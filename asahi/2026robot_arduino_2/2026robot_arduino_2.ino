// v6.12 STABILITY TELEMETRY TEST: mode 5 read-only rolling SD logging.
// v6.11 motor/align/orbit decisions preserved; verify on safe bench first.
// v6.7: v6.6 drive/ALIGN/orbit preserved; merged test_261029 Mode 2 and 3.
// Mode 2/3 angle signs follow v6.6 10-digit parsing, not test_261029.
// v6.4 120-mm orbit field build; original 10-digit wire parser preserved.
// 3-wheel omni robot + last-year style 10-digit command mode
//
// Last-year numeric command format (10 digits):
// [0]   modeq
// [1]   angle sign flag: 1=+, other=-
// [2:5] angle abs [deg] 000-999
// [5]   distance sign flag: 1=+, other=-
// [6:10] distance abs [mm] 0000-9999
//
// Example:
// 1100100200  -> mode=1, angle=+100 deg, distance=-200 mm
// 1000501000  -> mode=1, angle=-005 deg, distance=+1000 mm
//
// Mode map:
// 0: STOP
// 1: VISION_MOVE          angle priority -> distance correction, same idea as last year mode 1
// 2: CAMERA_MOVE          STS3215 camera servo via Mega Serial2
// 3: HIT_MODE_3           hitting servo + solenoid sequence
// 4: AUTO_ABSOLUTE        encoder-based rotate then move, similar role to last year mode 4
// 5: ALIGN_120 (four sensors)
// 6: ALIGN_120 -> continuous BALL_ORBIT -> ALIGN_120 (four sensors)
//
// Text commands are also left for bench testing:
// X 100 / Y 100 / ROT 30 / ARC 80 0 30 / SENSOR / ALIGN 499 / BALLROT 30 499 / STOP

#include <Arduino.h>
#include <math.h>
#include <Servo.h>
#include <SCServo.h>

// =====================
// Motor pins
// =====================
const int M1_IN1 = 5;
const int M1_IN2 = 6;

const int M2_IN1 = 7;
const int M2_IN2 = 8;

const int M3_IN1 = 9;
const int M3_IN2 = 10;

const int STBY = 4;

// =====================
// Encoder pins
// =====================
const int M1_ENC_A = 2;
const int M1_ENC_B = 22;

const int M2_ENC_A = 3;
const int M2_ENC_B = 23;

const int M3_ENC_A = 18;
const int M3_ENC_B = 24;

// =====================
// IR sensor pins
// =====================
const int IR_L_PIN = A0;
const int IR_R_PIN = A1;
const int IR_NEAR_L_PIN = A3;
const int IR_NEAR_R_PIN = A4;
const int IR_MEDIAN_COUNT = 15;  // must be odd

// =====================
// Mode 2: camera STS3215 bus servo (adjust for your hardware!)
// =====================
// Arduino Mega: Serial = Jetson USB 115200, Serial2 = STS3215 bus 1 Mbps
// Serial2 TX2=16, RX2=17. Use a suitable half-duplex TTL servo bus interface.
SMS_STS cameraServo;
const uint8_t CAMERA_SERVO_ID = 1;       // Change if actual servo ID differs
const unsigned long CAMERA_SERVO_BAUD = 1000000UL; // Match servo EEPROM baud

// '0 deg' means CAMERA_ZERO_POS encoder count, not an absolute homing operation.
// Check camera mounting, direction and safe mechanical range before running.
const int CAMERA_ZERO_POS = 2048;       // Calibrate camera forward/horizontal angle
const int CAMERA_DIRECTION = 1;        // +1 or -1: reverse if necessary
const int CAMERA_MIN_ANGLE_DEG = -65;  // Temporary conservative limit: calibrate
const int CAMERA_MAX_ANGLE_DEG =  90;  // Temporary conservative limit: calibrate
const int CAMERA_SERVO_SPEED = 1000;
const uint8_t CAMERA_SERVO_ACC = 200;
const int CAMERA_POS_TOLERANCE = 18;    // Encoder counts (~1.6 deg)
const uint8_t CAMERA_STABLE_READS = 3; // Require consecutive position matches
const unsigned long CAMERA_READ_INTERVAL_MS = 60UL;
const unsigned long CAMERA_TIMEOUT_MS = 12000UL;


// =====================
// Mode 3: hitting mechanism (standard PWM servo + solenoid)
// =====================
// Set false to dry-run with no actuator movement (DONE after 5 s).
// Set true ONLY after checking wiring, servo mechanical limits, and solenoid driver.
const bool HIT_MECHANISM_INSTALLED = true; // Completed hardware: Mode 3 enabled
const unsigned long HIT_SIMULATION_WAIT_MS = 5000UL;

// Dedicated pins; Mega's other motor pins (4-10), camera Serial2 (16/17),
// and encoder pins do not use these pins.
const uint8_t HIT_SERVO_PIN = 26;
const uint8_t HIT_SOLENOID_PIN = 27;  // MOSFET/driver input, NOT solenoid coil directly
const int HIT_HOME_ANGLE = 0;
const int HIT_READY_ANGLE = 120;
const int HIT_MIN_ANGLE = 35;
const int HIT_MAX_ANGLE = 140;

Servo hitServo;

// Preserve timing and operation order of the standalone test without delay().
enum HitStep {
  HIT_STEP_IDLE,
  HIT_WAIT_BEFORE_SET, // wait 1000 ms, then set striking servo angle
  HIT_WAIT_SET,        // wait 1000 ms, then release/strike
  HIT_WAIT_FIRE,       // wait 2000 ms, then return home
  HIT_WAIT_HOME,       // wait 1000 ms, then lock
  HIT_WAIT_LOCK,       // wait 1000 ms, then move to ready angle
  HIT_WAIT_READY       // wait 1000 ms, then report DONE
};
HitStep hitStep = HIT_STEP_IDLE;
unsigned long hitStepStartedAt = 0;
int hitTargetAngle = HIT_MIN_ANGLE;

// =====================
// Numeric command format
// =====================
const int EXPECTED_LEN = 10;

volatile int commandMode = 0;
volatile int visionAngleDeg = 0;
volatile int visionDistMm = 0;

// =====================
// Robot constants
// =====================
const float COUNTS_PER_REV = 1247.5;
const float WHEEL_DIAMETER_MM = 60.0;
const float WHEEL_CIRCUMFERENCE_MM = PI * WHEEL_DIAMETER_MM;
const float COUNTS_PER_MM = COUNTS_PER_REV / WHEEL_CIRCUMFERENCE_MM;
// Empirical ground-travel scaling on artificial turf; wheel rpm remains unscaled.
const float X_ODOM_SCALE = 0.678;
const float Y_ODOM_SCALE = 0.632;

const float L_WHEEL_MM = 140.0;

// =====================
// Control constants
// =====================
const float M1_GAIN = 1.00;
const float M2_GAIN = 1.00;
const float M3_GAIN = 1.00;

// ===== GRASS TEST: tune these values first =====
const float KP_RPM = 0.7;               // proportional rpm correction
const int GRASS_START_PWM = 155;        // starting boost (test at low load first)
const int GRASS_RUN_BASE_PWM = 112;     // running PWM before rpm correction
const float GRASS_RUN_PWM_PER_RPM = 2.0;
const int GRASS_FINE_PWM = 155;         // pulsed PWM in ALIGN, not continuous
const byte GRASS_FINE_ON_TICKS = 1;     // 1 x CONTROL_INTERVAL on
const byte GRASS_FINE_PERIOD_TICKS = 2; // 2 x CONTROL_INTERVAL off
const byte GRASS_ALIGN_STABLE_TICKS = 3; // sensor readings inside tolerance
// =================================================

const float M1_PWM_GAIN = 0.6;
const float M2_PWM_GAIN = 0.8;
const float M3_PWM_GAIN = 0.6;

const float PWM_STEP_LIMIT = 1.0;  // reserved. Not used yet.
const int DEADZONE_PWM = 40;
const int PWM_LIMIT = 255;

const unsigned long CONTROL_INTERVAL = 100;

// motion speed for encoder-based commands
const float MOVE_SPEED_MM_S = 110.0;
const float MOVE_SLOW_MM_S = 90.0;
const float ROT_SPEED_RAD_S = 0.35;
const float ROT_FINE_BAND_DEG = 30.0f; // ROT only: use fine pulses in final 10 degrees
const float ROT_FINE_OMEGA_RAD_S = 0.18f; // direction for pulsed final rotation
const float ARC_SPEED_RAD_S = 0.15;

// stepwise BALLROT setting
const float BALLROT_STEP_DEG = 30.0;      // rotate this many degrees, then ALIGN again
const float BALLROT_RADIUS_MM = 195.0;    // same virtual center offset as the old BALLROT

// motion speed for vision-style mode 1
const float VISION_MOVE_FAST_MM_S = 180.0;
const float VISION_MOVE_SLOW_MM_S = 120.0;
const float VISION_ROT_FAST_RAD_S = 0.35;
const float VISION_ROT_SLOW_RAD_S = 0.20;
const float VISION_DIST_SLOW_BAND_MM = 350.0;
const float VISION_ANGLE_SLOW_BAND_DEG = 8.0;

// stop tolerance
const float MOVE_TOL_MM = 5.0;
const float ROT_TOL_RAD = 3.0 * PI / 180.0;
const float VISION_DIST_TOL_MM = 10.0;
const float VISION_ANGLE_TOL_DEG = 3.0; // Mode 1: within 3 deg stops rotating

// sensor control (new IR sensors, RAW ADC units)
// Measured best position with 9-point median filter:
//   L ~ 502, R ~ 495, Avg ~ 499, Diff(L-R) ~ +7
// Four-sensor experimental baselines (recheck after driver replacement).
const float TARGET_120_MID_L = 474.0;
const float TARGET_120_MID_R = 470.0;
const float TARGET_160_MID_L = 374.0;
const float TARGET_160_MID_R = 380.0;
// 200 mm reference from stationary 4-sensor measurements (provisional).
const float TARGET_200_MID_L = 301.0;
const float TARGET_200_MID_R = 305.0;
const float ORBIT_START_SENSOR_DISTANCE_MM = 120.0f; // Mode 6 uses same target as Mode 5
const float MID_VALID_MIN_RAW = 185.0;  // PROVISIONAL: verify no-ball readings
// Separate centered-pair detection from single-side lateral recovery.
// 210 is a TEST threshold: static readings were ~224 at 300 mm, background ~150.
const float MID_CENTER_MIN_RAW = 190.0;
const float MID_SIDE_MIN_RAW = 200.0;   // strong single mid sensor -> lateral only
const float ALIGN_CENTER_DIFF_GATE = 35.0; // require near-centered pair before distance correction
const float ORBIT_PAIR_MIN_RAW = 350.0; // for 120mm: BOTH mids required; stop on weak detection
const float ALIGN_DIFF_TOL_RAW_4 = 14.0;
const float ALIGN_AVG_TOL_RAW_4 = 10.0;
const unsigned long ALIGN_TIMEOUT_MS = 40000UL;
const unsigned long ORBIT_TIMEOUT_MS = 120000UL; // independent safety timeout remains
const byte ALIGN_STABLE_READS = 3;
const byte SENSOR_LOST_TICKS = 5; // Stop immediately on lost data; search after debounce
// v6: conservative field-search limits (incrementing lateral displacement, mm)
const float SEARCH_STEPS_MM[] = {25.0f, -25.0f, 50.0f, -50.0f, 75.0f, -75.0f, 0.0f};
const byte SEARCH_STEP_COUNT = sizeof(SEARCH_STEPS_MM) / sizeof(SEARCH_STEPS_MM[0]);
const float SEARCH_SPEED_MM_S = 35.0f;
const unsigned long SEARCH_TIMEOUT_MS = 18000UL;
const float SEARCH_MAX_ABS_Y_MM = 80.0f;
const byte REACQUIRE_STABLE_READS = 3;
// Mode 5 only: in-place rotation replaces side translation for alignment.
// On prolonged sensor loss, perform a bounded rotational sweep. Mode 6 unchanged.
// v6.11 Mode 5 HYBRID: only large, sustained lateral errors cause in-place yaw.
// Mode 6 is deliberately excluded from all of these corrections.
const float MODE5_ROT_START_ERROR_RAW = 40.0f; // start turning after 3 confirmations
const float MODE5_ROT_STOP_ERROR_RAW = 20.0f;  // stop turning after 3 confirmations
const byte MODE5_ROT_CONFIRM_READS = 3;
const float MODE5_LATERAL_FINISH_BAND_RAW = 45.0f; // near 120mm: use side translation, not yaw
const float MODE5_MAX_HEADING_CORRECTION_DEG = 15.0f; // per attempt; then use lateral correction
const float MODE5_DIFF_FILTER_ALPHA = 0.25f; // low pass on paired MID difference
const float MODE5_ROT_EQUIV_RADIUS_MM = 180.0f; // converts former lateral speed into yaw rate
const float MODE5_ALIGN_OMEGA_MIN_RAD_S = 0.08f;
const float MODE5_ALIGN_OMEGA_MAX_RAD_S = 0.20f;
const float MODE5_HEADING_SCAN_TARGETS_DEG[] = {8.0f,-8.0f,16.0f,-16.0f,24.0f,-24.0f,0.0f};
const byte MODE5_HEADING_SCAN_STEP_COUNT = sizeof(MODE5_HEADING_SCAN_TARGETS_DEG)/sizeof(float);
const float MODE5_HEADING_SCAN_TOL_DEG = 2.5f;
const float MODE5_HEADING_SCAN_OMEGA_MAX_RAD_S = 0.22f;
const unsigned long MODE5_HEADING_SCAN_TIMEOUT_MS = 25000UL;
const byte MODE5_HEADING_SCAN_DWELL_TICKS = 2; // settle at each heading before measuring
const byte MODE5_HEADING_SCAN_VALID_TICKS = 2; // two consecutive pair detections before resuming ALIGN
const float MODE5_HEADING_SCAN_MIN_RAW = 205.0f;
// No centering-difference gate in scan; normal ALIGN decides when forward motion is safe.
const float MODE5_HEADING_SCAN_SAMPLE_JUMP_RAW = 80.0f; // allow changing signal during rotation
const float DIST_NEAR_BAND_MM = 32.0f; // Mode X/Y: last 32mm pulse
const float DIST_FINE_MIN_MM = 5.0f;
const float MODE1_FINE_DIST_MM = 35.0f; // Mode 1 distance <=35 mm: fine pulse
// ===== MODE 6 SITE TUNING (edit THESE fields before orbit test) =====
// IMPORTANT: x/y is BALL CENTER relative to ROBOT KINEMATIC CENTER.
// The 120mm sensor-to-ball distance is NOT this geometric x offset.
// Measure with ball at 120mm from the MID sensor reference face.
const bool ORBIT_SITE_GEOMETRY_MEASURED = true; // TRUE only after physical measurement & clearance check
const float ORBIT_SITE_BALL_X_MM = 260.0f;         // measured forward offset (+X) [mm]
const float ORBIT_SITE_BALL_Y_MM = 0.0f;         // measured left offset (+Y) [mm]
const float ORBIT_OMEGA_RAD_S = 0.08f;           // initial slow orbit; tune after 45-degree test
// No fixed maximum requested angle; orbit safety timeout and sensor-loss stop remain.
// ===== END SITE TUNING =====
const float ORBIT_GEOM_MIN_X_MM = 60.0f;
const float ORBIT_GEOM_MAX_X_MM = 600.0f;
// Existing debug ORBITCFG/ORBITARM can still be used; not necessary in normal operation.
// Defaults are loaded from the constants above at boot.
bool orbitGeometryConfirmed = ORBIT_SITE_GEOMETRY_MEASURED;
float orbitBallXmm = ORBIT_SITE_BALL_X_MM;
float orbitBallYmm = ORBIT_SITE_BALL_Y_MM;
const float ORBIT_CORR_MAX_MM_S = 8.0f; // conservative correction for 120mm
const float ORBIT_K_AVG = 0.10f;
const float ORBIT_K_DIFF = 0.08f;
const float ORBIT_ANGLE_TOL_RAD = 2.5f * PI / 180.0;
// Mid-sensor slope from measurements, approximate, only near centered ball.
const float MID_RAW_PER_MM = (472.0 - 377.0) / 40.0;

const float SENSOR_TARGET_AVG_RAW = 500.0;
const float SENSOR_TARGET_DIFF_RAW = 7.0;

// Provisional tolerances. DIFF tolerance is based on the current best-position data.
// AVG tolerance should be re-tuned after the front/back test.
const float SENSOR_AVG_TOL_RAW = 8.0;
const float SENSOR_DIFF_TOL_RAW = 14.0;

// Gains converted to RAW-count basis from the old voltage-based gains.
// 1 ADC count is about 5/1023 V. Start with these values and tune only if motion is too slow/fast.
const float SENSOR_K_AVG_RAW = 0.40;   // front-back correction mm/s per ADC count
const float SENSOR_K_DIFF_RAW = 0.15;  // left-right correction mm/s per ADC count (slower to avoid overshoot)

// Separate speed limits for front-back and left-right alignment.
const float SENSOR_V_LIMIT_AVG = 20.0;   // front-back max speed [mm/s]
const float SENSOR_V_LIMIT_DIFF = 10.0;  // left-right max speed [mm/s]

// Minimum front-back command while outside the AVG tolerance.
// Prevents the robot from stopping before it actually reaches the target.
const float SENSOR_V_MIN_AVG = 6.0;     // [mm/s]
const float SENSOR_V_MIN_DIFF = 6.0;  // 左右微調整の最低速度 [mm/s]

// Ball-detection threshold (provisional). Measure SENSOR with no ball and adjust if needed.
const float SENSOR_DETECT_RAW = 200.0;
const float ALIGN_SEARCH_RANGE_MM = 50.0;
const float ALIGN_SEARCH_SPEED_MM_S = 50.0;

// =====================
// Encoder counts
// =====================
volatile long countM1 = 0;
volatile long countM2 = 0;
volatile long countM3 = 0;

long startM1 = 0;
long startM2 = 0;
long startM3 = 0;

long prevM1 = 0;
long prevM2 = 0;
long prevM3 = 0;

// =====================
// PWM commands
// =====================
float pwmM1 = 0;
float pwmM2 = 0;
float pwmM3 = 0;
// Per-motor startup state for normal driving. Reset when a target becomes zero.
bool motorWasRequested[3] = {false, false, false};
// Shared pulse phase ensures all driven wheels use the same fine-control timing.
byte finePulsePhase = 0;
bool previousFineMode = false;
byte alignStableTicks = 0;

unsigned long prevTime = 0;

// =====================
// Mode
// =====================
enum Mode {
  IDLE,
  MOVE_X,
  MOVE_Y,
  ROT_CENTER,
  ROT_POINT,
  ALIGN_SEARCH_LEFT,
  ALIGN_SEARCH_RIGHT,
  ALIGN_BALL,
  BALL_ROT,             // old direct BALLROT slot. Not used by numeric mode 6 now.
  BALL_ROT_SEARCH_LEFT,
  BALL_ROT_SEARCH_RIGHT,
  BALL_ROT_ALIGN,
  BALL_ROT_STEP,
  VISION_MOVE,
  CAMERA_MOVE,       // mode 2: STS3215 camera servo
  HIT_MODE_3,        // mode 3: servo + solenoid striker
  AUTO_ABSOLUTE,     // mode 4: rotate then move using encoders
  ALIGN4,
  ORBIT_ALIGN_START,
  ORBIT_RUN,
  ORBIT_ALIGN120
};

Mode mode = IDLE;

float targetValue = 0.0;
float dirSign = 1.0;

float arcCx = 80.0;
float arcCy = 0.0;

float targetSensorRaw = SENSOR_TARGET_AVG_RAW;

// base command velocity
float cmdVx = 0.0;
float cmdVy = 0.0;
float cmdOmega = 0.0;

// AUTO_ABSOLUTE internal state
int autoAbsPhase = 0;  // 0=rotate, 1=move
float autoAbsDistanceMm = 0.0;

// Stepwise BALLROT internal state
float ballRotTotalRad = 0.0;
float ballRotDoneRad = 0.0;
float ballRotDirSign = 1.0;
float ballRotTargetSensorRaw = SENSOR_TARGET_AVG_RAW;

// Numeric one-shot/restart guard
int lastStartedNumericMode = -1;
float align4Distance = 120.0;
unsigned long phaseStartMs = 0;
byte fourStable = 0;
byte lostTicks = 0;
byte pulsePhase4 = 0;
float orbitTargetRad = 0;
float orbitSign = 1;
float orbitStartTheta = 0;
float orbitAccumRad = 0;
float orbitPrevTheta = 0;
// continuous wheel odometry: resetMotion() only changes origin for relative moves.
float totalTheta = 0;
float orbitLastVx = 0;
float orbitLastVy = 0;

int lastStartedAngleDeg = 0;
int lastStartedDistMm = 0;
// Search state exists only within ALIGN / pre-orbit ALIGN.
byte searchStep = 0;
bool searchActive = false;
byte reacquireCount = 0;
unsigned long searchStartMs = 0;
unsigned long lastAlignedSensorMs = 0;
bool motionFineOverride = false;
bool alignCoarseDrive = false;
// Mode 5 rotation-only reacquisition state; Mode 6 does not enter this scan.
bool mode5HeadingScanActive = false;
bool mode5HeadingScanAttempted = false;
byte mode5HeadingScanStep = 0;
byte mode5HeadingScanDwell = 0;
byte mode5HeadingScanStable = 0;
unsigned long mode5HeadingScanStartMs = 0;
float mode5HeadingPrevL = 0.0f, mode5HeadingPrevR = 0.0f;
bool mode5HeadingPrevValid = false;
// Cached readings come from the existing calculateAlign4() read (no extra ADC sampling).
bool mode5LastPair = false;
float mode5LastAvg = 0.0f, mode5LastDiffError = 0.0f;
bool mode5DiffFilterInitialized = false;
float mode5FilteredDiffError = 0.0f;
bool mode5RotCorrectionActive = false;
bool mode5RotCorrectionLimited = false;
byte mode5RotStartCount = 0, mode5RotStopCount = 0;
float mode5RotStartHeadingRad = 0.0f;

// =====================
// Mode 5: four-sensor rolling stability MONITOR (telemetry only).
// Sensor decisions and motor commands are deliberately unchanged.
// Each calculateAlign4() tick supplies one already-averaged/median RAW value.
// A 10-tick window at 100 ms nominal represents ~1 second while moving.
// SD<=5 is a stationary EXPERIMENTAL threshold, NOT a movement-quality gate.
// =====================
const byte MODE5_STAB_WINDOW = 10;
const float MODE5_STAB_SD_LIMIT = 5.0f;
float mode5StabSamples[4][MODE5_STAB_WINDOW];
byte mode5StabCount = 0;
byte mode5StabWrite = 0;
unsigned long mode5StabSeq = 0;

void resetMode5Stability() {
  mode5StabCount=0;
  mode5StabWrite=0;
  mode5StabSeq=0;
}

void printMode5Stability(const float raw[4]) {
  for (byte c=0;c<4;++c) mode5StabSamples[c][mode5StabWrite]=raw[c];
  mode5StabWrite=(mode5StabWrite+1)%MODE5_STAB_WINDOW;
  if (mode5StabCount<MODE5_STAB_WINDOW) ++mode5StabCount;
  ++mode5StabSeq;

  Serial.print(F("STAB5 ms=")); Serial.print(millis());
  Serial.print(F(" seq=")); Serial.print(mode5StabSeq);
  Serial.print(F(" n=")); Serial.print(mode5StabCount);
  for (byte c=0;c<4;++c) {
    Serial.print(c==0?F(" ML="):c==1?F(" MR="):c==2?F(" NL="):F(" NR="));
    Serial.print(raw[c],0);
    Serial.print(F(",sd="));
    if (mode5StabCount<MODE5_STAB_WINDOW) {
      Serial.print(F("NA,flag=WARMUP"));
      continue;
    }
    float mean=0.0f;
    for (byte i=0;i<MODE5_STAB_WINDOW;++i) mean+=mode5StabSamples[c][i];
    mean/=MODE5_STAB_WINDOW;
    float sumSq=0.0f;
    for (byte i=0;i<MODE5_STAB_WINDOW;++i) {
      const float e=mode5StabSamples[c][i]-mean;
      sumSq+=e*e;
    }
    const float sd=sqrtf(sumSq/(MODE5_STAB_WINDOW-1)); // sample SD
    Serial.print(sd,2);
    Serial.print(F(",flag="));
    Serial.print(sd<=MODE5_STAB_SD_LIMIT?F("STABLE"):F("UNSTABLE"));
  }
  Serial.println();
}

// Common adaptive pulse is still bounded by CONTROL_INTERVAL (100ms).


struct FourSensors {
  float midL, midR, nearL, nearR, avg, diff;
};
FourSensors readFourSensors();
bool centralPairValid(const FourSensors &f);
bool orbitPairValid(const FourSensors &f);
void failFour(const char* reason);
void finishControlTick(long c1,long c2,long c3,unsigned long now);
void startAlign4(float distance,bool fromOrbit);
void startOrbit4(float angleDeg);
bool calculateAlign4(float distance,float &vx,float &vy,bool &valid);
void startMode5HeadingScan(unsigned long now);
void runMode5HeadingScan(unsigned long now,float theta_est,float rpm1,float rpm2,float rpm3,
                         long c1,long c2,long c3);

// Mode 2 state: nonblocking position feedback with timeout.
bool cameraMoveActive = false;
int cameraTargetPos = 2048;
unsigned long cameraMoveStartedAt = 0;
unsigned long cameraLastReadAt = 0;
uint8_t cameraStableCount = 0;

void startCameraMove(int angleDeg);
void updateCameraMove();

// Mode 3 state: independent of wheel encoder control.
bool hitMode3Active = false;
unsigned long hitMode3StartedAt = 0;

void startHitMode3(float angleDeg, float distanceMm);
void updateHitMode3();
void startRealHit();
bool isRealHitDone();
void cancelHitMode3();

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  Serial2.begin(CAMERA_SERVO_BAUD); // Mega TX2 D16, RX2 D17
  cameraServo.pSerial = &Serial2;
  // Keep solenoid in its safe LOCK state at boot. No automatic firing.
  pinMode(HIT_SOLENOID_PIN, OUTPUT);
  digitalWrite(HIT_SOLENOID_PIN, LOW);
  if (HIT_MECHANISM_INSTALLED) {
    hitServo.attach(HIT_SERVO_PIN);
    hitServo.write(HIT_READY_ANGLE); // 120°
  }


  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);
  pinMode(M3_IN1, OUTPUT);
  pinMode(M3_IN2, OUTPUT);
  pinMode(STBY, OUTPUT);

  pinMode(M1_ENC_A, INPUT_PULLUP);
  pinMode(M1_ENC_B, INPUT_PULLUP);
  pinMode(M2_ENC_A, INPUT_PULLUP);
  pinMode(M2_ENC_B, INPUT_PULLUP);
  pinMode(M3_ENC_A, INPUT_PULLUP);
  pinMode(M3_ENC_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(M1_ENC_A), readEncoderM1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(M2_ENC_A), readEncoderM2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(M3_ENC_A), readEncoderM3, CHANGE);

  digitalWrite(STBY, HIGH);

  prevTime = millis();

  Serial.println(F("Ready"));
  Serial.println(F("Numeric command: mode + sign/angle + sign/distance = 10 digits"));
  Serial.println(F("Mode 0 STOP / 1 VISION_MOVE / 2 CAMERA_MOVE / 3 HIT_MODE_3 / 4 AUTO_ABSOLUTE / 5 ALIGN_120 / 6 ORBIT_120_THEN_ALIGN_120"));
  Serial.println(F("Text: X/Y/ROT/ALIGN 120/ALIGN 160/ALIGN 200/BALLROT 90/SENSOR/ORBITCFG x y/ORBITARM/ORBITDISARM/STOP"));
}


// =====================
// Shared sensor ALIGN control
// Used by both standalone ALIGN (mode 5) and BALLROT internal ALIGN (mode 6).
// This keeps both modes on exactly the same alignment logic and tuning values.
// =====================
void readSensorRawValues(float &rawL, float &rawR, float &rawAvg, float &rawDiff);

bool calculateAlignControl(float targetRaw,
                           float &outVx, float &outVy,
                           float &rawL, float &rawR,
                           float &rawAvg, float &rawDiff) {
  readSensorRawValues(rawL, rawR, rawAvg, rawDiff);

  float errorAvg = targetRaw - rawAvg;
  float errorDiff = SENSOR_TARGET_DIFF_RAW - rawDiff;

  outVx = 0.0;
  outVy = 0.0;

  // Both front/back and left/right are inside the allowed range.
  if (abs(errorAvg) <= SENSOR_AVG_TOL_RAW &&
      abs(errorDiff) <= SENSOR_DIFF_TOL_RAW) {
    if (alignStableTicks < GRASS_ALIGN_STABLE_TICKS) alignStableTicks++;
    return (alignStableTicks >= GRASS_ALIGN_STABLE_TICKS);
  }
  alignStableTicks = 0;

  // Give left/right correction priority while DIFF is outside its tolerance.
  if (abs(errorDiff) > SENSOR_DIFF_TOL_RAW) {
    outVy = -SENSOR_K_DIFF_RAW * errorDiff;

    // Keep enough command to overcome static friction during fine correction.
    if (abs(outVy) < SENSOR_V_MIN_DIFF) {
      outVy = (outVy >= 0.0) ? SENSOR_V_MIN_DIFF : -SENSOR_V_MIN_DIFF;
    }

    outVy = constrain(outVy,
                      -SENSOR_V_LIMIT_DIFF,
                       SENSOR_V_LIMIT_DIFF);
  }
  else {
    // Left/right is acceptable, so correct front/back position.
    outVx = SENSOR_K_AVG_RAW * errorAvg;

    // Keep enough command to overcome static friction near the target.
    if (abs(outVx) < SENSOR_V_MIN_AVG) {
      outVx = (outVx >= 0.0) ? SENSOR_V_MIN_AVG : -SENSOR_V_MIN_AVG;
    }

    outVx = constrain(outVx,
                      -SENSOR_V_LIMIT_AVG,
                       SENSOR_V_LIMIT_AVG);
  }

  return false;
}

void loop() {
  readLatestCommand();
  updateCameraMove();
  updateHitMode3();

  unsigned long now = millis();

  if (now - prevTime >= CONTROL_INTERVAL) {
    controlLoop(now);
  }
}

// =====================
// Serial command
// =====================
bool isNumeric10(String line) {
  if (line.length() != EXPECTED_LEN) return false;
  for (uint8_t i = 0; i < line.length(); i++) {
    if (!isDigit(line[i])) return false;
  }
  return true;
}

void readLatestCommand() {
  if (!Serial.available()) return;

  String last = "";

  // Same idea as last year's code: use only the newest complete command.
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      last = line;
    }
  }

  if (last.length() == 0) return;

  if (isNumeric10(last)) {
    parseNumericCommand(last);
  } else {
    parseTextCommand(last);
  }
}

bool isSameNumericCommand(int modeVal, int angleDeg, int distMm) {
  return (lastStartedNumericMode == modeVal &&
          lastStartedAngleDeg == angleDeg &&
          lastStartedDistMm == distMm);
}

void rememberStartedNumericCommand(int modeVal, int angleDeg, int distMm) {
  lastStartedNumericMode = modeVal;
  lastStartedAngleDeg = angleDeg;
  lastStartedDistMm = distMm;
}

void clearStartedNumericCommand() {
  lastStartedNumericMode = -1;
  lastStartedAngleDeg = 0;
  lastStartedDistMm = 0;
}

bool isBallRotStepwiseMode() {
  return (mode == BALL_ROT_SEARCH_LEFT ||
          mode == BALL_ROT_SEARCH_RIGHT ||
          mode == BALL_ROT_ALIGN ||
          mode == BALL_ROT_STEP);
}

void parseNumericCommand(String line) {
  int modeVal = line.substring(0, 1).toInt();

  int signFlagDeg = line.substring(1, 2).toInt();
  int angleAbs = line.substring(2, 5).toInt();
  int angleSigned = (signFlagDeg == 1) ? -angleAbs : angleAbs;

  int signFlagDis = line.substring(5, 6).toInt();
  int disAbs = line.substring(6, 10).toInt();
  int disSigned = (signFlagDis == 1) ? disAbs : -disAbs;

  // Cancel Mode 2/3 safely when a different numeric mode arrives.
  if (modeVal != 2) cameraMoveActive = false;
  if (modeVal != 3 && hitMode3Active) cancelHitMode3();
  commandMode = modeVal;
  visionAngleDeg = angleSigned;
  visionDistMm = disSigned;

  Serial.print(F("Numeric received mode="));
  Serial.print(modeVal);
  Serial.print(F(" angle="));
  Serial.print(angleSigned);
  Serial.print(F(" dist="));
  Serial.println(disSigned);

  switch (modeVal) {
    case 0:
      stopAll();
      mode = IDLE;
      clearStartedNumericCommand();
      Serial.println(F("Mode 0: STOP"));
      break;

    case 1:
      // Mode 1 is a continuous vision-style command.
      // The newest angle/distance values are used without restarting every time.
      if (mode != VISION_MOVE) {
        resetMotion();
      }
      mode = VISION_MOVE;
      rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      break;

    case 2:
      // Angle field controls STS3215. Distance field is ignored.
      // Guard against repeated 10-digit commands after DONE.
      if (!isSameNumericCommand(modeVal, angleSigned, disSigned)) {
        startCameraMove(angleSigned);
        rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      }
      break;

    case 3:
      // The distance field selects the hitting-servo angle (0..100 mm mapped to 35..140 deg).
      if (hitMode3Active) {
        Serial.println(F("HIT_BUSY: mode 3 command ignored"));
      } else if (!isSameNumericCommand(modeVal, angleSigned, disSigned)) {
        startHitMode3((float)angleSigned, (float)disSigned);
        rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      }
      break;

    case 4:
      // Encoder-based one-shot command. Do not restart while the same command is repeatedly sent.
      if (!isSameNumericCommand(modeVal, angleSigned, disSigned) || mode != AUTO_ABSOLUTE) {
        startAutoAbsolute((float)angleSigned, (float)disSigned);
        rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      }
      break;

    case 5:
      // Preserve 10-digit wire format: all 10 bytes interpreted as before.
      // Distance field remains parsed but Mode 5 is fixed at 120 mm.
      if (!isSameNumericCommand(modeVal, angleSigned, disSigned)) {
        startAlign4(120.0, false);
        rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      }
      break;

    case 6:
      // Angle field uses the exact original sign conversion, distance remains ignored.
      if (!isSameNumericCommand(modeVal, angleSigned, disSigned)) {
        startOrbit4((float)angleSigned);
        rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      }
      break;

    default:
      stopAll();
      mode = IDLE;
      clearStartedNumericCommand();
      Serial.println(F("Unknown numeric mode. STOP."));
      break;
  }
}

void parseTextCommand(String line) {
  // Diagnostic SENSOR does not interrupt actuators. All other textual commands do.
  if (line != "SENSOR") {
    cameraMoveActive = false;
    if (hitMode3Active) cancelHitMode3();
  }
  Serial.print(F("Received: ["));
  Serial.print(line);
  Serial.println(F("]"));

  if (line == "STOP") {
    stopAll();
    mode = IDLE;
    clearStartedNumericCommand();
    Serial.println(F("STOP"));
    return;
  }

  if (line == "SENSOR") { printSensor(); return; }
  if (line == "ORBITDISARM") {
    orbitGeometryConfirmed = false; stopAll(); mode = IDLE;
    Serial.println(F("ORBIT_DISARMED")); return;
  }
  if (line == "ORBITARM") {
    if (orbitBallXmm < ORBIT_GEOM_MIN_X_MM || orbitBallXmm > ORBIT_GEOM_MAX_X_MM || abs(orbitBallYmm)>150.0f) {
      Serial.println(F("ERROR:ORBIT_GEOMETRY_INVALID")); return;
    }
    orbitGeometryConfirmed = true;
    Serial.println(F("ORBIT_ARMED_CHECK_CLEARANCE")); return;
  }
  if (line.startsWith("ORBITCFG ")) {
    // Test command only. x/y are robot-center to BALL CENTER, not sensor distance.
    int sp = line.indexOf(' ', 9);
    if (sp < 0) { Serial.println(F("ERROR:ORBITCFG_USAGE")); return; }
    orbitBallXmm = line.substring(9,sp).toFloat();
    orbitBallYmm = line.substring(sp+1).toFloat();
    orbitGeometryConfirmed = false;
    Serial.print(F("ORBIT_CONFIG X=")); Serial.print(orbitBallXmm);
    Serial.print(F(" Y=")); Serial.println(orbitBallYmm);
    return;
  }

  int sp1 = line.indexOf(' ');
  String cmd;

  if (sp1 < 0) {
    cmd = line;
  } else {
    cmd = line.substring(0, sp1);
  }

  if (cmd == "X") {
    float mm = getArg(line, 1);
    startMoveX(mm);
  }
  else if (cmd == "Y") {
    float mm = getArg(line, 1);
    startMoveY(mm);
  }
  else if (cmd == "ROT") {
    float deg = getArg(line, 1);
    startRotateCenter(deg);
  }
  else if (cmd == "ARC") {
    float x = getArg(line, 1);
    float y = getArg(line, 2);
    float deg = getArg(line, 3);
    startRotatePoint(x, y, deg);
  }
  else if (cmd == "ALIGN") {
    float distance = getArg(line, 1);
    if (distance == 0.0f) distance = 120.0f;
    startAlign4(distance, false);
  }
  else if (cmd == "BALLROT" || cmd == "ORBIT") {
    float deg = getArg(line, 1);
    startOrbit4(deg);
  }
  else {
    Serial.println(F("Unknown command"));
  }
}

float getArg(String line, int index) {
  line.trim();

  int start = 0;
  int currentIndex = 0;

  while (true) {
    int space = line.indexOf(' ', start);

    String token;

    if (space < 0) {
      token = line.substring(start);
    } else {
      token = line.substring(start, space);
    }

    token.trim();

    if (currentIndex == index) {
      return token.toFloat();
    }

    if (space < 0) break;

    start = space + 1;
    currentIndex++;
  }

  return 0.0;
}

// =====================
// Start modes
// =====================
void resetMotion() {
  noInterrupts();
  startM1 = countM1;
  startM2 = countM2;
  startM3 = countM3;

  prevM1 = countM1;
  prevM2 = countM2;
  prevM3 = countM3;
  interrupts();

  pwmM1 = 0;
  pwmM2 = 0;
  pwmM3 = 0;

  prevTime = millis();
}

void startMoveX(float mm) {
  resetMotion();

  mode = MOVE_X;
  targetValue = abs(mm);
  dirSign = (mm >= 0) ? 1.0 : -1.0;

  cmdVx = MOVE_SPEED_MM_S * dirSign;
  cmdVy = 0.0;
  cmdOmega = 0.0;

  Serial.print(F("Start X "));
  Serial.println(mm);
}

void startMoveY(float mm) {
  resetMotion();

  mode = MOVE_Y;
  targetValue = abs(mm);
  dirSign = (mm >= 0) ? 1.0 : -1.0;

  cmdVx = 0.0;
  cmdVy = MOVE_SPEED_MM_S * dirSign;
  cmdOmega = 0.0;

  Serial.print(F("Start Y "));
  Serial.println(mm);
}

void startRotateCenter(float deg) {
  resetMotion();

  mode = ROT_CENTER;
  targetValue = abs(deg) * PI / 180.0;
  dirSign = (deg >= 0) ? 1.0 : -1.0;

  cmdVx = 0.0;
  cmdVy = 0.0;
  cmdOmega = ROT_SPEED_RAD_S * dirSign;

  Serial.print(F("Start ROT "));
  Serial.println(deg);
}

void startRotatePoint(float x, float y, float deg) {
  resetMotion();

  mode = ROT_POINT;
  arcCx = x;
  arcCy = y;

  targetValue = abs(deg) * PI / 180.0;
  dirSign = (deg >= 0) ? 1.0 : -1.0;

  cmdOmega = ARC_SPEED_RAD_S * dirSign;

  // Translation velocity for rotation around point (cx, cy)
  cmdVx = cmdOmega * arcCy;
  cmdVy = -cmdOmega * arcCx;

  Serial.print(F("Start ARC cx="));
  Serial.print(x);
  Serial.print(F(", cy="));
  Serial.print(y);
  Serial.print(F(", deg="));
  Serial.println(deg);
}

void startAlign(float targetRaw) {
  resetMotion();

  targetSensorRaw = targetRaw;

  float rawL, rawR, rawAvg, rawDiff;
  readSensorRawValues(rawL, rawR, rawAvg, rawDiff);

  if (rawL <= SENSOR_DETECT_RAW && rawR <= SENSOR_DETECT_RAW) {
    mode = ALIGN_SEARCH_LEFT;
    targetValue = ALIGN_SEARCH_RANGE_MM;
    dirSign = 1.0;

    Serial.println(F("ALIGN: no object detected. Search left."));
  } else {
    mode = ALIGN_BALL;
    Serial.print(F("Start ALIGN targetRaw="));
    Serial.println(targetRaw, 1);
  }
}

void startBallRot(float deg, float targetRaw) {
  // Stepwise BALLROT:
  // 1) ALIGN targetRaw. If the ball is not detected during search, finish.
  // 2) Rotate a small angle.
  // 3) ALIGN again.
  // 4) Repeat until the requested total angle is reached.
  resetMotion();

  targetSensorRaw = targetRaw;
  ballRotTargetSensorRaw = targetRaw;
  ballRotTotalRad = abs(deg) * PI / 180.0;
  ballRotDoneRad = 0.0;
  ballRotDirSign = (deg >= 0) ? 1.0 : -1.0;

  float rawL, rawR, rawAvg, rawDiff;
  readSensorRawValues(rawL, rawR, rawAvg, rawDiff);

  if (rawL <= SENSOR_DETECT_RAW && rawR <= SENSOR_DETECT_RAW) {
    mode = BALL_ROT_SEARCH_LEFT;
    targetValue = ALIGN_SEARCH_RANGE_MM;
    dirSign = 1.0;

    Serial.print(F("Start BALLROT_STEPWISE deg="));
    Serial.print(deg);
    Serial.println(F(". First ALIGN: no object detected. Search left."));
  } else {
    mode = BALL_ROT_ALIGN;

    Serial.print(F("Start BALLROT_STEPWISE deg="));
    Serial.print(deg);
    Serial.print(F(", targetRaw="));
    Serial.print(targetRaw, 1);
    Serial.println(F(". First ALIGN."));
  }
}

void startCameraMove(int angleDeg) {
  // Hold wheels stopped and command the camera to an absolute angle relative to ZERO_POS.
  stopAll();
  mode = CAMERA_MOVE;
  cameraMoveActive = false;
  cameraStableCount = 0;

  // Reject uncalibrated / mechanically unsafe commands, rather than silently clamp.
  if (angleDeg < CAMERA_MIN_ANGLE_DEG || angleDeg > CAMERA_MAX_ANGLE_DEG) {
    mode = IDLE;
    Serial.println(F("CAMERA_ERROR: ANGLE_OUT_OF_RANGE"));
    return;
  }

  long target = (long)CAMERA_ZERO_POS
              + (long)CAMERA_DIRECTION * lroundf((float)angleDeg * 4096.0f / 360.0f);
  if (target < 0 || target > 4095) {
    mode = IDLE;
    Serial.println(F("CAMERA_ERROR: POSITION_OUT_OF_RANGE"));
    return;
  }

  // Verify feedback communication before issuing the motion command.
  int presentPos = cameraServo.ReadPos(CAMERA_SERVO_ID);
  if (presentPos < 0) {
    mode = IDLE;
    Serial.println(F("CAMERA_ERROR: NO_SERVO_FEEDBACK"));
    return;
  }

  cameraTargetPos = (int)target;
  cameraServo.WritePosEx(CAMERA_SERVO_ID, cameraTargetPos,
                         CAMERA_SERVO_SPEED, CAMERA_SERVO_ACC);
  cameraMoveStartedAt = millis();
  cameraLastReadAt = millis();
  cameraMoveActive = true;

  Serial.print(F("CAMERA_MOVE angle="));
  Serial.print(angleDeg);
  Serial.print(F(" targetPos="));
  Serial.print(cameraTargetPos);
  Serial.print(F(" startPos="));
  Serial.println(presentPos);
}

void updateCameraMove() {
  if (!cameraMoveActive) return;
  if (mode != CAMERA_MOVE) {
    cameraMoveActive = false;
    return;
  }

  unsigned long now = millis();
  if ((unsigned long)(now - cameraMoveStartedAt) >= CAMERA_TIMEOUT_MS) {
    cameraMoveActive = false;
    mode = IDLE;
    Serial.println(F("CAMERA_TIMEOUT")); // Jetson must not start ball detection
    return;
  }

  if ((unsigned long)(now - cameraLastReadAt) < CAMERA_READ_INTERVAL_MS) return;
  cameraLastReadAt = now;

  // ReadPos() returns actual encoder position, -1 on communication failure.
  int presentPos = cameraServo.ReadPos(CAMERA_SERVO_ID);
  if (presentPos < 0) {
    cameraStableCount = 0;
    Serial.println(F("CAMERA_WARN: READ_POS_FAILED"));
    return;
  }

  int errorPos = abs(cameraTargetPos - presentPos);
  if (errorPos <= CAMERA_POS_TOLERANCE) {
    if (cameraStableCount < CAMERA_STABLE_READS) cameraStableCount++;
  } else {
    cameraStableCount = 0;
  }

  if (cameraStableCount >= CAMERA_STABLE_READS) {
    cameraMoveActive = false;
    mode = IDLE;
    Serial.print(F("CAMERA_REACHED pos="));
    Serial.print(presentPos);
    Serial.print(F(" target="));
    Serial.println(cameraTargetPos);
    Serial.println(F("DONE")); // Exactly one completion notification to Jetson
  }
}

void startHitMode3(float angleDeg, float distanceMm) {
  // Wheels must stay stopped throughout the hit sequence.
  stopAll();
  mode = HIT_MODE_3;
  hitMode3StartedAt = millis();
  hitMode3Active = true;

  // Same conversion as the supplied standalone test:
  // signed distance [mm] clamped to 0..100, mapped to servo angle 35..140 deg.
  // NOTE: Jetson currently sends 0 mm (no FLAG) or 2000 mm (with FLAG),
  // which produces the endpoint angles 35 or 140 degrees respectively.
  int boundedDistMm = constrain((int)lroundf(distanceMm), 0, 100);
  hitTargetAngle = map(boundedDistMm, 0, 100, HIT_MIN_ANGLE, HIT_MAX_ANGLE);

  Serial.print(F("Mode 3 HIT requested. angle="));
  Serial.print(angleDeg);
  Serial.print(F(" dist="));
  Serial.print(distanceMm);
  Serial.print(F(" hitServoTarget="));
  Serial.println(hitTargetAngle);

  if (HIT_MECHANISM_INSTALLED) {
    startRealHit();
    Serial.println(F("Mode 3 REAL: hitting sequence started."));
  } else {
    Serial.println(F("Mode 3 SIMULATION: no hit. DONE after 5 seconds."));
  }
}

void updateHitMode3() {
  if (!hitMode3Active) return;

  // Do not emit a stale DONE if another mode or STOP has interrupted.
  if (mode != HIT_MODE_3) {
    cancelHitMode3();
    return;
  }

  if (HIT_MECHANISM_INSTALLED) {
    if (!isRealHitDone()) return;
  } else {
    if ((unsigned long)(millis() - hitMode3StartedAt) < HIT_SIMULATION_WAIT_MS) return;
  }

  hitMode3Active = false;
  hitStep = HIT_STEP_IDLE;
  stopAll();
  mode = IDLE;
  Serial.println(F("DONE")); // Jetson expects the exact string DONE, not 1.
}

void startRealHit() {
  // Start the same steps as the standalone hit tester, but use millis().
  // Keep LOW before initiating so the solenoid is locked outside the hit phase.
  digitalWrite(HIT_SOLENOID_PIN, LOW);
  hitStep = HIT_WAIT_BEFORE_SET;
  hitStepStartedAt = millis();
}

bool isRealHitDone() {
  unsigned long now = millis();
  unsigned long elapsed = (unsigned long)(now - hitStepStartedAt);

  switch (hitStep) {
    case HIT_WAIT_BEFORE_SET:
      if (elapsed < 1000UL) return false;
      hitServo.write(hitTargetAngle);  // Set servo target
      hitStep = HIT_WAIT_SET;
      break;

    case HIT_WAIT_SET:
      if (elapsed < 1000UL) return false;
      digitalWrite(HIT_SOLENOID_PIN, HIGH); // Strike (via driver)
      hitServo.write(hitTargetAngle - 2);
      hitStep = HIT_WAIT_FIRE;
      break;

    case HIT_WAIT_FIRE:
      if (elapsed < 3000UL) return false;
      hitServo.write(HIT_HOME_ANGLE);
      hitStep = HIT_WAIT_HOME;
      break;

    case HIT_WAIT_HOME:
      if (elapsed < 4000UL) return false;
      digitalWrite(HIT_SOLENOID_PIN, LOW); // Lock
      hitStep = HIT_WAIT_LOCK;
      break;

    case HIT_WAIT_LOCK:
      if (elapsed < 1000UL) return false;
      hitServo.write(HIT_READY_ANGLE);
      hitStep = HIT_WAIT_READY;
      break;

    case HIT_WAIT_READY:
      if (elapsed < 1000UL) return false;
      hitStep = HIT_STEP_IDLE;
      return true;

    default:
      return false;
  }

  hitStepStartedAt = now;
  return false;
}

void cancelHitMode3() {
  // Do not leave the solenoid energized if STOP or another mode arrives.
  // Confirm the mechanical safe position of the servo on the real robot.
  hitMode3Active = false;
  hitStep = HIT_STEP_IDLE;
  digitalWrite(HIT_SOLENOID_PIN, LOW); // LOCK on cancellation
  if (HIT_MECHANISM_INSTALLED) hitServo.write(HIT_HOME_ANGLE);
  Serial.println(F("HIT_CANCELLED"));
}

void startAutoAbsolute(float angleDeg, float distanceMm) {
  resetMotion();

  mode = AUTO_ABSOLUTE;
  autoAbsPhase = 0;
  autoAbsDistanceMm = distanceMm;

  targetValue = abs(angleDeg) * PI / 180.0;
  dirSign = (angleDeg >= 0) ? 1.0 : -1.0;

  Serial.print(F("Mode 4 AUTO_ABSOLUTE angle="));
  Serial.print(angleDeg);
  Serial.print(F(" dist="));
  Serial.println(distanceMm);
}

// =====================
// Main control loop
// =====================
void controlLoop(unsigned long now) {
  long c1, c2, c3;

  noInterrupts();
  c1 = countM1;
  c2 = countM2;
  c3 = countM3;
  interrupts();

  float dt = (now - prevTime) / 1000.0;
  if (dt <= 0.0) dt = CONTROL_INTERVAL / 1000.0;

  long d1 = c1 - prevM1;
  long d2 = c2 - prevM2;
  long d3 = c3 - prevM3;

  float rpm1 = (d1 / COUNTS_PER_REV) * (60.0 / dt);
  float rpm2 = (d2 / COUNTS_PER_REV) * (60.0 / dt);
  float rpm3 = (d3 / COUNTS_PER_REV) * (60.0 / dt);

  float s1 = (c1 - startM1) / COUNTS_PER_MM;
  float s2 = (c2 - startM2) / COUNTS_PER_MM;
  float s3 = (c3 - startM3) / COUNTS_PER_MM;

  float x_est = (s3 - s1) / 1.732 * X_ODOM_SCALE;
  float y_est = (s1 + s3 - 2.0 * s2) / 3.0 * Y_ODOM_SCALE;
  float theta_est = (s1 + s2 + s3) / (3.0 * L_WHEEL_MM);

  // Update integrated heading before mode-specific early returns.
  totalTheta += (d1 + d2 + d3) / (3.0f * COUNTS_PER_MM * L_WHEEL_MM);
  if (mode == IDLE || mode == CAMERA_MOVE || mode == HIT_MODE_3) {
    prevM1 = c1;
    prevM2 = c2;
    prevM3 = c3;
    prevTime = now;
    return;
  }

  float Vx = 0.0;
  float Vy = 0.0;
  float omega = 0.0;

  bool done = false;

  motionFineOverride = false; alignCoarseDrive = false;
  if (mode == ALIGN4 || mode == ORBIT_ALIGN_START || mode == ORBIT_ALIGN120) {
    const float distance = (mode == ORBIT_ALIGN_START) ? ORBIT_START_SENSOR_DISTANCE_MM :
                           (mode == ORBIT_ALIGN120) ? 120.0f :
                           align4Distance;
    // Only Mode 5 gets heading reacquisition. Mode 6 follows its original logic.
    if (mode == ALIGN4 && distance == 120.0f && mode5HeadingScanActive) {
      runMode5HeadingScan(now,theta_est,rpm1,rpm2,rpm3,c1,c2,c3);
      return;
    }
    bool valid = false;
    bool aligned = calculateAlign4(distance, Vx, Vy, valid);
    // Do not drive blindly during sensor loss; bounded progressively wider lateral scan.
    if (!valid) {
      Vx = 0; Vy = 0;
      fourStable = 0; reacquireCount = 0;
      if (!searchActive && ++lostTicks >= SENSOR_LOST_TICKS) {
        // Mode 5: search by rotating around the robot center, never by side translation.
        // An initial pair that is merely noisy does NOT trigger an extra scan.
        if (mode == ALIGN4 && distance == 120.0f) {
          if (!mode5HeadingScanAttempted) {
            startMode5HeadingScan(now);
          } else {
            failFour("MODE5_BALL_LOST");
          }
          finishControlTick(c1,c2,c3,now);
          return;
        }
        // Mode 6 keeps its original bounded side-translation reacquisition.
        searchActive = true; searchStep = 0; reacquireCount = 0;
        searchStartMs = now; resetMotion();
        Serial.println(F("SEARCH_BEGIN"));
      }
    } else {
      lostTicks = 0;
      if (searchActive) {
        if (++reacquireCount >= REACQUIRE_STABLE_READS) {
          searchActive = false; reacquireCount = 0;
          Serial.println(F("SEARCH_REACQUIRED"));
        } else { Vx=0; Vy=0; }
      }
    }
    if (searchActive) {
      // Search must not use distance data while uncertain. Start from stopped origin.
      if (now - searchStartMs >= SEARCH_TIMEOUT_MS || searchStep >= SEARCH_STEP_COUNT) {
        failFour("SEARCH_TIMEOUT"); finishControlTick(c1,c2,c3,now); return;
      }
      if (reacquireCount == 0) {
        float dy = SEARCH_STEPS_MM[searchStep] - y_est;
        if (abs(y_est) > SEARCH_MAX_ABS_Y_MM) {
          failFour("SEARCH_BOUNDARY"); finishControlTick(c1,c2,c3,now); return;
        }
        if (abs(dy) <= 6.0f) {
          stopAll(); ++searchStep; Vx=0; Vy=0;
          Serial.print(F("SEARCH_STEP=")); Serial.println(searchStep);
        } else {
          Vx=0; Vy=(dy>0?SEARCH_SPEED_MM_S:-SEARCH_SPEED_MM_S);
        }
      }
    }
    // Mode 5 ONLY: distant + sustained large lateral error -> in-place heading correction.
    // Small fluctuations -> forward drive; within 45 RAW of target distance ->
    // existing sensor-based fine lateral correction. Mode 6 logic stays untouched.
    if (mode == ALIGN4 && distance == 120.0f && !searchActive) {
      const float targetAvg120 = 0.5f * (TARGET_120_MID_L + TARGET_120_MID_R);
      const float errorAvg120 = targetAvg120 - mode5LastAvg;
      const bool nearTarget = mode5LastPair &&
                        (errorAvg120 <= MODE5_LATERAL_FINISH_BAND_RAW);
      if (!valid) {
        mode5RotCorrectionActive = false;
        mode5RotStartCount = mode5RotStopCount = 0;
        mode5DiffFilterInitialized = false;
      } else if (nearTarget || mode5RotCorrectionLimited) {
        // Finish with lateral translation; never rotate close to the ball.
        mode5RotCorrectionActive = false;
        mode5RotStartCount = mode5RotStopCount = 0;
        // Vx/Vy remain exactly as calculated by calculateAlign4().
      } else if (mode5LastPair) {
        const float e = mode5LastDiffError;  // targetDiff - measuredDiff
        if (!mode5DiffFilterInitialized) {
          mode5FilteredDiffError = e;
          mode5DiffFilterInitialized = true;
        } else {
          mode5FilteredDiffError += MODE5_DIFF_FILTER_ALPHA *
                                      (e - mode5FilteredDiffError);
        }
        const float absErr = abs(mode5FilteredDiffError);
        if (mode5RotCorrectionActive) {
          if (absErr <= MODE5_ROT_STOP_ERROR_RAW) {
            if (mode5RotStopCount < MODE5_ROT_CONFIRM_READS) ++mode5RotStopCount;
          } else mode5RotStopCount = 0;
          if (mode5RotStopCount >= MODE5_ROT_CONFIRM_READS) {
            mode5RotCorrectionActive = false;
            mode5RotStartCount = mode5RotStopCount = 0;
            Serial.println(F("MODE5_HEADING_ALIGNED"));
          } else if (abs((totalTheta - mode5RotStartHeadingRad) * 180.0f / PI)
                      >= MODE5_MAX_HEADING_CORRECTION_DEG) {
            mode5RotCorrectionActive = false;
            mode5RotCorrectionLimited = true;
            Serial.println(F("MODE5_HEADING_LIMIT_LATERAL_FALLBACK"));
          }
        } else if (absErr >= MODE5_ROT_START_ERROR_RAW) {
          if (mode5RotStartCount < MODE5_ROT_CONFIRM_READS) ++mode5RotStartCount;
          if (mode5RotStartCount >= MODE5_ROT_CONFIRM_READS) {
            mode5RotCorrectionActive = true;
            mode5RotStartHeadingRad = totalTheta;
            mode5RotStopCount = 0;
            Serial.println(F("MODE5_HEADING_CORRECT_BEGIN"));
          }
        } else mode5RotStartCount = 0;

        if (mode5RotCorrectionActive) {
          // Former Vy sign convention preserved; filtered error is target - actual.
          const float vyEquivalent = constrain(-SENSOR_K_DIFF_RAW * mode5FilteredDiffError,
                                                -SENSOR_V_LIMIT_DIFF, SENSOR_V_LIMIT_DIFF);
          float w = constrain(vyEquivalent / MODE5_ROT_EQUIV_RADIUS_MM,
                              -MODE5_ALIGN_OMEGA_MAX_RAD_S, MODE5_ALIGN_OMEGA_MAX_RAD_S);
          if (abs(w) < MODE5_ALIGN_OMEGA_MIN_RAD_S)
            w = (w >= 0.0f) ? MODE5_ALIGN_OMEGA_MIN_RAD_S : -MODE5_ALIGN_OMEGA_MIN_RAD_S;
          omega = w; Vx = 0.0f; Vy = 0.0f;
          alignCoarseDrive = false;
        } else if (absErr > ALIGN_CENTER_DIFF_GATE &&
                   !mode5RotCorrectionLimited) {
          // Wait for the three-sample decision rather than making an impulse move.
          Vx = Vy = 0.0f;
        } else if (!mode5RotCorrectionLimited) {
          // Coarse approach: forward translation takes precedence over a small
          // lateral correction. Reuse the original ALIGN velocity constants.
          Vy = 0.0f;
          alignCoarseDrive = (errorAvg120 > 55.0f);
          if (abs(errorAvg120) <= ALIGN_AVG_TOL_RAW_4) Vx = 0.0f;
          else if (alignCoarseDrive) Vx = 65.0f;
          else {
            Vx = constrain(SENSOR_K_AVG_RAW * errorAvg120,
                           -SENSOR_V_LIMIT_AVG, SENSOR_V_LIMIT_AVG);
            if (abs(Vx) < SENSOR_V_MIN_AVG)
              Vx = (Vx >= 0.0f) ? SENSOR_V_MIN_AVG : -SENSOR_V_MIN_AVG;
          }
        }
      } else {
        // Only one MID sensor sees the ball: preserve lateral-only recovery.
        // Never infer distance or yaw from an unreliable paired difference.
        mode5RotStartCount = mode5RotStopCount = 0;
        mode5DiffFilterInitialized = false;
        mode5RotCorrectionActive = false;
      }
    }
    if (mode == ALIGN4) {
      Serial.print(F("CMD5 ms=")); Serial.print(millis());
      Serial.print(F(" valid=")); Serial.print(valid?1:0);
      Serial.print(F(" vx=")); Serial.print(Vx,1);
      Serial.print(F(" vy=")); Serial.print(Vy,1);
      Serial.print(F(" omega=")); Serial.print(omega,3);
      Serial.print(F(" coarse=")); Serial.print(alignCoarseDrive?1:0);
      Serial.print(F(" rotate=")); Serial.println(mode5RotCorrectionActive?1:0);
    }
    if (millis() - phaseStartMs > ALIGN_TIMEOUT_MS) {
      failFour("ALIGN_TIMEOUT"); finishControlTick(c1,c2,c3,now); return;
    }
    if (aligned) {
      stopAll();
      if (mode == ALIGN4) {
        mode = IDLE;
        Serial.println(F("DONE"));
      } else if (mode == ORBIT_ALIGN_START) {
        if (!orbitGeometryConfirmed || orbitBallXmm <= 0.0f) {
          failFour("GEOMETRY_NOT_CONFIGURED");
        } else {
          mode = ORBIT_RUN;
          phaseStartMs = millis();
          orbitAccumRad = 0.0f;
          orbitPrevTheta = totalTheta;
          Serial.println(F("ORBIT_START"));
        }
      } else {
        mode = IDLE;
        Serial.println(F("DONE"));
      }
      finishControlTick(c1,c2,c3,now);
      return;
    }
    // Only use pulse control when correcting near target; after each ON tick
    // the OFF ticks let the sensors settle before another pulse.
    pulsePhase4 = (pulsePhase4 + 1) % GRASS_FINE_PERIOD_TICKS;
  }
  else if (mode == ORBIT_RUN) {
    FourSensors fs = readFourSensors();
    const float deltaTheta = totalTheta - orbitPrevTheta;
    orbitPrevTheta = totalTheta;
    orbitAccumRad += orbitSign * deltaTheta;
    bool valid = orbitPairValid(fs);
    if (!valid) {
      if (++lostTicks >= 3) { failFour("SENSOR_LOST"); finishControlTick(c1,c2,c3,now); return; }
      // Immediate halt rather than blindly following nominal orbit.
      stopAll(); finishControlTick(c1,c2,c3,now); return;
    } else lostTicks = 0;
    if (millis() - phaseStartMs > min(ORBIT_TIMEOUT_MS, (unsigned long)(10000.0f + 1500.0f * orbitTargetRad / ORBIT_OMEGA_RAD_S))) {
      failFour("ORBIT_TIMEOUT"); finishControlTick(c1,c2,c3,now); return;
    }
    if (orbitAccumRad >= orbitTargetRad - ORBIT_ANGLE_TOL_RAD) {
      stopAll();
      mode = ORBIT_ALIGN120;
      phaseStartMs = millis(); fourStable = 0; lostTicks = 0; pulsePhase4 = 0;
      Serial.println(F("ORBIT_DONE ALIGN_120_START"));
      finishControlTick(c1,c2,c3,now);
      return;
    }
    omega = orbitSign * ORBIT_OMEGA_RAD_S;
    Vx = omega * orbitBallYmm;
    Vy = -omega * orbitBallXmm;
    // Correction only while both center sensors are valid; otherwise briefly
    // hold the nominal orbit, then abort after consecutive invalid readings.
    if (valid) {
      float targetAvg = (TARGET_120_MID_L + TARGET_120_MID_R) * 0.5f;
      float targetDiff = TARGET_120_MID_L - TARGET_120_MID_R;
      Vx += constrain(ORBIT_K_AVG * (targetAvg - fs.avg), -ORBIT_CORR_MAX_MM_S, ORBIT_CORR_MAX_MM_S);
      Vy += constrain(ORBIT_K_DIFF * (fs.diff - targetDiff), -ORBIT_CORR_MAX_MM_S, ORBIT_CORR_MAX_MM_S);
    }
    Serial.print(F("ORBIT angle=")); Serial.print(orbitAccumRad * 180.0f / PI,1);
    Serial.print(F(" mid=")); Serial.print(fs.midL); Serial.print(','); Serial.println(fs.midR);
  }
  else if (mode == VISION_MOVE) {
    computeVisionMoveCommand(Vx, Vy, omega);
    // Mode 1 uses the same pulse driver as the local ROT and X/Y moves.
    // Rotation has priority: 3 < |angle| <= 10 deg -> pulse rotation.
    // Only once angle is settled: 10 < |distance| <= 35 mm -> pulse X.
    const float angleErrorAbs = abs(visionAngleDeg);
    const float distanceErrorAbs = abs(visionDistMm);
    const bool visionRotFine = (angleErrorAbs > VISION_ANGLE_TOL_DEG &&
                                angleErrorAbs <= ROT_FINE_BAND_DEG);
    const bool visionDistFine = (angleErrorAbs <= VISION_ANGLE_TOL_DEG &&
                                 distanceErrorAbs > VISION_DIST_TOL_MM &&
                                 distanceErrorAbs <= MODE1_FINE_DIST_MM);
    motionFineOverride = visionRotFine || visionDistFine;
    done = false;  // Keep this mode alive until mode 0 or another command arrives.
  }
  else if (mode == MOVE_X) {
    float progress = dirSign * x_est;
    float remain = targetValue - progress;

    if (remain <= MOVE_TOL_MM) {
      done = true;
    } else {
      float speed = (remain < DIST_NEAR_BAND_MM) ? MOVE_SLOW_MM_S : MOVE_SPEED_MM_S;
      Vx = speed * dirSign;
      motionFineOverride = (remain < DIST_NEAR_BAND_MM);
    }
  }
  else if (mode == MOVE_Y) {
    float progress = dirSign * y_est;
    float remain = targetValue - progress;

    if (remain <= MOVE_TOL_MM) {
      done = true;
    } else {
      float speed = (remain < DIST_NEAR_BAND_MM) ? MOVE_SLOW_MM_S : MOVE_SPEED_MM_S;
      Vy = speed * dirSign;
      motionFineOverride = (remain < DIST_NEAR_BAND_MM);
    }
  }
  else if (mode == ROT_CENTER) {
    float progress = dirSign * theta_est;
    float remain = targetValue - progress;

    if (remain <= ROT_TOL_RAD) {
      done = true;
    } else {
      // ROT only: coarse continuous rotation, then 100ms ON / 100ms OFF pulses.
      // The existing ROT_TOL_RAD (3 deg) still defines completion.
      bool rotFine = (remain <= ROT_FINE_BAND_DEG * PI / 180.0f);
      omega = (rotFine ? ROT_FINE_OMEGA_RAD_S : ROT_SPEED_RAD_S) * dirSign;
      motionFineOverride = rotFine;
    }
  }
  else if (mode == ROT_POINT) {
    float progress = dirSign * theta_est;
    float remain = targetValue - progress;

    if (remain <= ROT_TOL_RAD) {
      done = true;
    } else {
      omega = (remain < 0.25) ? 0.15 * dirSign : ARC_SPEED_RAD_S * dirSign;

      Vx = omega * arcCy;
      Vy = -omega * arcCx;
    }
  }
  else if (mode == ALIGN_SEARCH_LEFT) {
    float vL, vR, vAvg, vDiff;
    readSensorRawValues(vL, vR, vAvg, vDiff);

    float progress = y_est;

    if (vL > SENSOR_DETECT_RAW || vR > SENSOR_DETECT_RAW) {
      resetMotion();
      mode = ALIGN_BALL;
      Serial.println(F("ALIGN: object found. Start normal ALIGN."));
    }
    else if (progress >= ALIGN_SEARCH_RANGE_MM) {
      resetMotion();
      mode = ALIGN_SEARCH_RIGHT;
      targetValue = ALIGN_SEARCH_RANGE_MM * 2.0;
      Serial.println(F("ALIGN: not found left. Search right."));
    }
    else {
      Vx = 0.0;
      Vy = ALIGN_SEARCH_SPEED_MM_S;
      omega = 0.0;
    }

    Serial.print(F("SEARCH_LEFT AvgRaw="));
    Serial.print(vAvg, 1);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == ALIGN_SEARCH_RIGHT) {
    float vL, vR, vAvg, vDiff;
    readSensorRawValues(vL, vR, vAvg, vDiff);

    float progress = -y_est;

    if (vL > SENSOR_DETECT_RAW || vR > SENSOR_DETECT_RAW) {
      resetMotion();
      mode = ALIGN_BALL;
      Serial.println(F("ALIGN: object found. Start normal ALIGN."));
    }
    else if (progress >= ALIGN_SEARCH_RANGE_MM * 2.0) {
      stopAll();
      mode = IDLE;
      done = true;
      Serial.println(F("ALIGN: object not found. Finish."));
    }
    else {
      Vx = 0.0;
      Vy = -ALIGN_SEARCH_SPEED_MM_S;
      omega = 0.0;
    }

    Serial.print(F("SEARCH_RIGHT AvgRaw="));
    Serial.print(vAvg, 1);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == ALIGN_BALL) {
    float vL, vR, vAvg, vDiff;

    bool aligned = calculateAlignControl(targetSensorRaw,
                                         Vx, Vy,
                                         vL, vR, vAvg, vDiff);
    omega = 0.0;

    if (aligned) {
      done = true;
    }

    Serial.print(F("ALIGN | Lraw="));
    Serial.print(vL, 1);
    Serial.print(F(" Rraw="));
    Serial.print(vR, 1);
    Serial.print(F(" AvgRaw="));
    Serial.print(vAvg, 1);
    Serial.print(F(" DiffRaw="));
    Serial.print(vDiff, 1);
    Serial.print(F(" Vx="));
    Serial.print(Vx);
    Serial.print(F(" Vy="));
    Serial.println(Vy);
  }
  else if (mode == BALL_ROT_SEARCH_LEFT) {
    float vL, vR, vAvg, vDiff;
    readSensorRawValues(vL, vR, vAvg, vDiff);

    float progress = y_est;

    if (vL > SENSOR_DETECT_RAW || vR > SENSOR_DETECT_RAW) {
      resetMotion();
      mode = BALL_ROT_ALIGN;
      Serial.println(F("BALLROT: object found. ALIGN before step."));

      prevM1 = c1;
      prevM2 = c2;
      prevM3 = c3;
      prevTime = now;
      return;
    }
    else if (progress >= ALIGN_SEARCH_RANGE_MM) {
      resetMotion();
      mode = BALL_ROT_SEARCH_RIGHT;
      targetValue = ALIGN_SEARCH_RANGE_MM * 2.0;
      Serial.println(F("BALLROT: not found left. Search right."));

      prevM1 = c1;
      prevM2 = c2;
      prevM3 = c3;
      prevTime = now;
      return;
    }
    else {
      Vx = 0.0;
      Vy = ALIGN_SEARCH_SPEED_MM_S;
      omega = 0.0;
    }

    Serial.print(F("BALLROT_SEARCH_LEFT AvgRaw="));
    Serial.print(vAvg, 1);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == BALL_ROT_SEARCH_RIGHT) {
    float vL, vR, vAvg, vDiff;
    readSensorRawValues(vL, vR, vAvg, vDiff);

    float progress = -y_est;

    if (vL > SENSOR_DETECT_RAW || vR > SENSOR_DETECT_RAW) {
      resetMotion();
      mode = BALL_ROT_ALIGN;
      Serial.println(F("BALLROT: object found. ALIGN before step."));

      prevM1 = c1;
      prevM2 = c2;
      prevM3 = c3;
      prevTime = now;
      return;
    }
    else if (progress >= ALIGN_SEARCH_RANGE_MM * 2.0) {
      stopAll();
      mode = IDLE;
      done = true;
      Serial.println(F("BALLROT: object not found. Finish without rotation."));
    }
    else {
      Vx = 0.0;
      Vy = -ALIGN_SEARCH_SPEED_MM_S;
      omega = 0.0;
    }

    Serial.print(F("BALLROT_SEARCH_RIGHT AvgRaw="));
    Serial.print(vAvg, 1);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == BALL_ROT_ALIGN) {
    float vL, vR, vAvg, vDiff;

    // Use exactly the same ALIGN control as standalone mode 5.
    bool aligned = calculateAlignControl(ballRotTargetSensorRaw,
                                         Vx, Vy,
                                         vL, vR, vAvg, vDiff);
    omega = 0.0;

    if (aligned) {
      float remainingTotal = ballRotTotalRad - ballRotDoneRad;

      if (remainingTotal <= ROT_TOL_RAD) {
        // This is the final re-ALIGN after the last rotation step.
        done = true;
        Serial.println(F("BALLROT: final ALIGN done."));
      }
      else {
        // Initial ALIGN or an intermediate re-ALIGN is complete.
        // Proceed to the next rotation step.
        float stepRad = BALLROT_STEP_DEG * PI / 180.0;
        if (stepRad > remainingTotal) stepRad = remainingTotal;

        resetMotion();
        mode = BALL_ROT_STEP;
        targetValue = stepRad;
        dirSign = ballRotDirSign;

        Serial.print(F("BALLROT: ALIGN done. Start step deg="));
        Serial.print(stepRad * 180.0 / PI);
        Serial.print(F(" / remaining deg="));
        Serial.println(remainingTotal * 180.0 / PI);

        prevM1 = c1;
        prevM2 = c2;
        prevM3 = c3;
        prevTime = now;
        return;
      }
    }

    Serial.print(F("BALLROT_ALIGN | Lraw="));
    Serial.print(vL, 1);
    Serial.print(F(" Rraw="));
    Serial.print(vR, 1);
    Serial.print(F(" AvgRaw="));
    Serial.print(vAvg, 1);
    Serial.print(F(" DiffRaw="));
    Serial.print(vDiff, 1);
    Serial.print(F(" Vx="));
    Serial.print(Vx);
    Serial.print(F(" Vy="));
    Serial.print(Vy);
    Serial.print(F(" doneDeg="));
    Serial.println(ballRotDoneRad * 180.0 / PI);
  }
  else if (mode == BALL_ROT_STEP) {
    float progress = dirSign * theta_est;
    float remain = targetValue - progress;

    if (remain <= ROT_TOL_RAD) {
      ballRotDoneRad += targetValue;

      float remainingTotal = ballRotTotalRad - ballRotDoneRad;

      if (remainingTotal <= ROT_TOL_RAD) {
        // Requested rotation is complete, but do not finish yet.
        // Perform one final ALIGN, then BALL_ROT_ALIGN will set done=true.
        resetMotion();
        mode = BALL_ROT_ALIGN;

        Serial.print(F("BALLROT: final step done. Final Re-ALIGN. doneDeg="));
        Serial.println(ballRotDoneRad * 180.0 / PI);

        prevM1 = c1;
        prevM2 = c2;
        prevM3 = c3;
        prevTime = now;
        return;
      } else {
        resetMotion();
        mode = BALL_ROT_ALIGN;

        Serial.print(F("BALLROT: small step done. Re-ALIGN. doneDeg="));
        Serial.print(ballRotDoneRad * 180.0 / PI);
        Serial.print(F(" remainingDeg="));
        Serial.println(remainingTotal * 180.0 / PI);

        prevM1 = c1;
        prevM2 = c2;
        prevM3 = c3;
        prevTime = now;
        return;
      }
    } else {
      float vL, vR, vAvg, vDiff;
      readSensorRawValues(vL, vR, vAvg, vDiff);

      float errorAvg = ballRotTargetSensorRaw - vAvg;
      float errorDiff = SENSOR_TARGET_DIFF_RAW - vDiff;

      omega = (remain < 0.08) ? 0.15 * dirSign : ARC_SPEED_RAD_S * dirSign;

      Vx = constrain(SENSOR_K_AVG_RAW * errorAvg,
                     -SENSOR_V_LIMIT_AVG,
                      SENSOR_V_LIMIT_AVG);
      Vy = -constrain(SENSOR_K_DIFF_RAW * errorDiff,
                      -SENSOR_V_LIMIT_DIFF,
                       SENSOR_V_LIMIT_DIFF);

      // Rotation around a virtual point near the ball.
      Vx += omega * 0.0;
      Vy += -omega * BALLROT_RADIUS_MM;

      Serial.print(F("BALLROT_STEP IR AvgRaw="));
      Serial.print(vAvg, 1);
      Serial.print(F(", Diff="));
      Serial.print(vDiff, 1);
      Serial.print(F(", stepTheta="));
      Serial.print(theta_est * 180.0 / PI);
      Serial.print(F(", doneDeg="));
      Serial.println(ballRotDoneRad * 180.0 / PI);
    }
  }
  else if (mode == AUTO_ABSOLUTE) {
    if (autoAbsPhase == 0) {
      float progress = dirSign * theta_est;
      float remain = targetValue - progress;

      if (remain <= ROT_TOL_RAD || targetValue <= ROT_TOL_RAD) {
        if (abs(autoAbsDistanceMm) > MOVE_TOL_MM) {
          resetMotion();
          mode = AUTO_ABSOLUTE;
          autoAbsPhase = 1;
          targetValue = abs(autoAbsDistanceMm);
          dirSign = (autoAbsDistanceMm >= 0) ? 1.0 : -1.0;

          prevM1 = c1;
          prevM2 = c2;
          prevM3 = c3;
          prevTime = now;
          return;
        } else {
          done = true;
        }
      } else {
        omega = (remain < 0.25) ? 0.18 * dirSign : ROT_SPEED_RAD_S * dirSign;
      }
    }
    else {
      float progress = dirSign * x_est;
      float remain = targetValue - progress;

      if (remain <= MOVE_TOL_MM) {
        done = true;
      } else {
        float speed = (remain < DIST_NEAR_BAND_MM) ? MOVE_SLOW_MM_S : MOVE_SPEED_MM_S;
        Vx = speed * dirSign;
        motionFineOverride = (remain < DIST_NEAR_BAND_MM);
      }
    }
  }

  if (done) {
    stopAll();
    mode = IDLE;

    Serial.print(F("Done. x="));
    Serial.print(x_est);
    Serial.print(F(", y="));
    Serial.print(y_est);
    Serial.print(F(", th="));
    Serial.println(theta_est * 180.0 / PI);

    // Mode 5 completion notification
    if (commandMode == 5) {
      Serial.println(F("DONE"));
    }


    prevM1 = c1;
    prevM2 = c2;
    prevM3 = c3;
    prevTime = now;
    return;
  }

  driveRobot(Vx, Vy, omega, rpm1, rpm2, rpm3);

  Serial.print(F("mode="));
  Serial.print((int)mode);
  Serial.print(F(", x="));
  Serial.print(x_est);
  Serial.print(F(", y="));
  Serial.print(y_est);
  Serial.print(F(", th="));
  Serial.print(theta_est * 180.0 / PI);
  Serial.print(F(", V="));
  Serial.print(Vx);
  Serial.print(F(","));
  Serial.print(Vy);
  Serial.print(F(","));
  Serial.print(omega);
  Serial.print(F(", rpm="));
  Serial.print(rpm1);
  Serial.print(F(","));
  Serial.print(rpm2);
  Serial.print(F(","));
  Serial.println(rpm3);

  prevM1 = c1;
  prevM2 = c2;
  prevM3 = c3;
  prevTime = now;
}

void computeVisionMoveCommand(float &Vx, float &Vy, float &omega) {
  Vx = 0.0;
  Vy = 0.0;
  omega = 0.0;

  int eDeg = visionAngleDeg;
  int dMm = visionDistMm;

  // 1) Angle first, like last year's computeCommandRates().
  if (abs(eDeg) > VISION_ANGLE_TOL_DEG) {
    // Coarse rotation above 10 deg, pulse rotation within 10 deg.
    float w = (abs(eDeg) > ROT_FINE_BAND_DEG) ? VISION_ROT_FAST_RAD_S : ROT_FINE_OMEGA_RAD_S;
    omega = (eDeg >= 0) ? w : -w;
    return;
  }

  // 2) Distance after angle is roughly aligned.
  if (abs(dMm) > VISION_DIST_TOL_MM) {
    float speed = (abs(dMm) > VISION_DIST_SLOW_BAND_MM) ? VISION_MOVE_FAST_MM_S : VISION_MOVE_SLOW_MM_S;
    Vx = (dMm >= 0) ? speed : -speed;  // + means forward in robot X direction.
    return;
  }

  // 3) Settled.
  Vx = 0.0;
  Vy = 0.0;
  omega = 0.0;
}

// =====================
// Drive calculation
// =====================
void driveRobot(float Vx, float Vy, float omega, float rpm1, float rpm2, float rpm3) {
  float v1 = -0.866 * Vx + 0.5 * Vy + L_WHEEL_MM * omega;
  float v2 = -Vy + L_WHEEL_MM * omega;
  float v3 =  0.866 * Vx + 0.5 * Vy + L_WHEEL_MM * omega;

  float targetRpm1 = (v1 / WHEEL_CIRCUMFERENCE_MM * 60.0) * M1_GAIN;
  float targetRpm2 = (v2 / WHEEL_CIRCUMFERENCE_MM * 60.0) * M2_GAIN;
  float targetRpm3 = (v3 / WHEEL_CIRCUMFERENCE_MM * 60.0) * M3_GAIN;

  // Fine pulse drive is enabled by near-target motionFineOverride (X/Y/ROT/Mode 1)
  // or by the existing sensor-based ALIGN modes.
  bool fine = motionFineOverride || (mode == ALIGN_BALL || mode == BALL_ROT_ALIGN || ((mode == ALIGN4 || mode == ORBIT_ALIGN_START || mode == ORBIT_ALIGN120) && !alignCoarseDrive && !searchActive));
  if (fine) {
    if (!previousFineMode) finePulsePhase = 0;
  } else {
    finePulsePhase = 0;
  }
  previousFineMode = fine;
  bool pulseOn = (finePulsePhase < GRASS_FINE_ON_TICKS);
  if (fine) finePulsePhase = (finePulsePhase + 1) % GRASS_FINE_PERIOD_TICKS;

  // 微調整パルスのPWM倍率
  const float FINE_Y_FACTOR   = 0.85f;  // 左右移動
  const float FINE_ROT_FACTOR = 0.6f;  // 回転

  float fineFactor = 1.0f;

  if (fine) {
    if (abs(omega) > 0.001f &&
        abs(Vx) < 0.1f && abs(Vy) < 0.1f) {
      fineFactor = FINE_ROT_FACTOR;
    }
    else if (abs(Vy) > 0.1f &&
            abs(Vx) < 0.1f && abs(omega) < 0.001f) {
      fineFactor = FINE_Y_FACTOR;
    }
  }

  updateMotorGrass(targetRpm1, rpm1, pwmM1,
                  M1_PWM_GAIN * fineFactor, 0, fine, pulseOn);

  updateMotorGrass(targetRpm2, rpm2, pwmM2,
                  M2_PWM_GAIN * fineFactor, 1, fine, pulseOn);

  updateMotorGrass(targetRpm3, rpm3, pwmM3,
                  M3_PWM_GAIN * fineFactor, 2, fine, pulseOn);

  setMotor(M1_IN1, M1_IN2, (int)pwmM1);
  setMotor(M2_IN1, M2_IN2, (int)pwmM2);
  setMotor(M3_IN1, M3_IN2, (int)pwmM3);
}

void updateMotorGrass(float targetRpm, float currentRpm, float &pwm,
                      float gain, byte motorIndex, bool fine, bool pulseOn) {
  if (abs(targetRpm) < 0.5) {
    pwm = 0;
    motorWasRequested[motorIndex] = false;
    return;
  }
  float sign = (targetRpm >= 0.0) ? 1.0 : -1.0;
  if (fine) {
    // Pulsed low-speed operation. Sensor feedback determines when to stop;
    // encoder RPM is intentionally not fed back during the short pulses.
    pwm = pulseOn ? sign * constrain(GRASS_FINE_PWM * gain, 0.0f, (float)PWM_LIMIT) : 0;
    motorWasRequested[motorIndex] = false;
    return;
  }

  float base = GRASS_RUN_BASE_PWM + GRASS_RUN_PWM_PER_RPM * abs(targetRpm);
  float correction = KP_RPM * (abs(targetRpm) - sign * currentRpm);
  float demanded = (base + correction) * gain;
  if (!motorWasRequested[motorIndex]) {
    demanded = max(demanded, (float)GRASS_START_PWM * gain);
  }
  pwm = sign * constrain(demanded, 0.0f, (float)PWM_LIMIT);
  motorWasRequested[motorIndex] = true;
}

// =====================
// Motor output
// =====================
void setMotor(int in1, int in2, int pwm) {
  if (pwm > 0) {
    int out = pwm;

    if (out < DEADZONE_PWM) out = DEADZONE_PWM;
    if (out > 255) out = 255;

    analogWrite(in1, out);
    analogWrite(in2, 0);
  }
  else if (pwm < 0) {
    int out = -pwm;

    if (out < DEADZONE_PWM) out = DEADZONE_PWM;
    if (out > 255) out = 255;

    analogWrite(in1, 0);
    analogWrite(in2, out);
  }
  else {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
  }
}

void stopAll() {
  pwmM1 = 0;
  pwmM2 = 0;
  pwmM3 = 0;
  for (byte i = 0; i < 3; i++) motorWasRequested[i] = false;
  previousFineMode = false;
  finePulsePhase = 0;
  alignStableTicks = 0;

  analogWrite(M1_IN1, 255);
  analogWrite(M1_IN2, 255);

  analogWrite(M2_IN1, 255);
  analogWrite(M2_IN2, 255);

  analogWrite(M3_IN1, 255);
  analogWrite(M3_IN2, 255);
}

// =====================
// Sensor (new IR sensors)
// =====================
void sortIntArray(int data[], int n) {
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (data[j] < data[i]) {
        int t = data[i];
        data[i] = data[j];
        data[j] = t;
      }
    }
  }
}

void readSensorRawValues(float &rawL, float &rawR, float &rawAvg, float &rawDiff) {
  int left[IR_MEDIAN_COUNT];
  int right[IR_MEDIAN_COUNT];

  // Interleave L/R reads so both medians represent nearly the same time window.
  // A dummy read after channel switching reduces ADC carry-over between A0 and A1.
  for (int i = 0; i < IR_MEDIAN_COUNT; i++) {
    analogRead(IR_L_PIN);
    delayMicroseconds(100);
    left[i] = analogRead(IR_L_PIN);

    analogRead(IR_R_PIN);
    delayMicroseconds(100);
    right[i] = analogRead(IR_R_PIN);

    delay(2);
  }

  sortIntArray(left, IR_MEDIAN_COUNT);
  sortIntArray(right, IR_MEDIAN_COUNT);

  rawL = left[IR_MEDIAN_COUNT / 2];
  rawR = right[IR_MEDIAN_COUNT / 2];
  rawAvg = (rawL + rawR) / 2.0;
  rawDiff = rawL - rawR;
}

void printSensor() {
  FourSensors f=readFourSensors();
  Serial.print(F("MID_L:"));Serial.print(f.midL,1);
  Serial.print(F(" MID_R:"));Serial.print(f.midR,1);
  Serial.print(F(" NEAR_L:"));Serial.print(f.nearL,1);
  Serial.print(F(" NEAR_R:"));Serial.print(f.nearR,1);
  Serial.print(F(" AVG:"));Serial.print(f.avg,1);
  Serial.print(F(" DIFF:"));Serial.println(f.diff,1);
  return;
}


// =============================================================
// 4-sensor alignment and continuous orbit (2026-11-02 spec).
// Sensor reference: MID A0/A1, NEAR A3/A4.
// v6 bounded lateral search in ALIGN modes only; ORBIT aborts on lost track.
// =============================================================
float readMedianPin(byte pin) {
  int a[IR_MEDIAN_COUNT];
  for (byte i=0; i<IR_MEDIAN_COUNT; ++i) {
    analogRead(pin); delayMicroseconds(80);
    a[i] = analogRead(pin);
  }
  sortIntArray(a,IR_MEDIAN_COUNT);
  return (float)a[IR_MEDIAN_COUNT/2];
}

FourSensors readFourSensors() {
  FourSensors f;
  f.midL=readMedianPin(IR_L_PIN);
  f.midR=readMedianPin(IR_R_PIN);
  f.nearL=readMedianPin(IR_NEAR_L_PIN);
  f.nearR=readMedianPin(IR_NEAR_R_PIN);
  f.avg=0.5f*(f.midL+f.midR);
  f.diff=f.midL-f.midR;
  return f;
}

bool centralPairValid(const FourSensors &f) {
  // Both sides required for distance estimation; background must be checked at test site.
  return (f.midL >= MID_CENTER_MIN_RAW && f.midR >= MID_CENTER_MIN_RAW);
}

bool orbitPairValid(const FourSensors &f) {
  // Orbit remains conservative even when ALIGN acquisition is broadened.
  return f.midL >= ORBIT_PAIR_MIN_RAW && f.midR >= ORBIT_PAIR_MIN_RAW;
}

void failFour(const char* reason) {
  stopAll();
  mode = IDLE;

  Serial.print(F("ERROR:"));
  Serial.println(reason);

  Serial.println(F("DONE"));  // エラー通知後に完了信号を送る
}

void finishControlTick(long c1,long c2,long c3,unsigned long now) {
  prevM1=c1; prevM2=c2; prevM3=c3; prevTime=now;
}

void startAlign4(float distance, bool fromOrbit) {
  if (distance != 120.0f && distance != 160.0f && distance != 200.0f) {
    failFour("ALIGN_DISTANCE_ONLY_120_160_200"); return;
  }
  stopAll(); resetMotion();
  mode = fromOrbit ? ORBIT_ALIGN_START : ALIGN4;
  align4Distance = distance;
  if (!fromOrbit) resetMode5Stability();
  phaseStartMs=millis(); fourStable=0; lostTicks=0; pulsePhase4=0; searchActive=false; searchStep=0; reacquireCount=0;
  mode5HeadingScanActive=false; mode5HeadingScanAttempted=false;
  mode5RotCorrectionActive=false; mode5RotCorrectionLimited=false;
  mode5RotStartCount=0; mode5RotStopCount=0;
  mode5DiffFilterInitialized=false;
  Serial.print(F("ALIGN4_START distance=")); Serial.println(distance,0);
}

void startOrbit4(float angleDeg) {
  // Do not spin an unknown radius or use the old 195-mm virtual orbit point.
  if (!orbitGeometryConfirmed || orbitBallXmm < ORBIT_GEOM_MIN_X_MM ||
      orbitBallXmm > ORBIT_GEOM_MAX_X_MM || abs(orbitBallYmm) > 150.0f) {
    failFour("GEOMETRY_NOT_CONFIGURED"); return;
  }
  if (!isfinite(angleDeg) || angleDeg == 0.0f) {
    failFour("INVALID_ORBIT_ANGLE"); return;
  }
  orbitSign=(angleDeg >= 0.0f)?1.0f:-1.0f;
  orbitTargetRad=abs(angleDeg)*PI/180.0f;
  stopAll(); resetMotion();
  mode=ORBIT_ALIGN_START;
  phaseStartMs=millis(); fourStable=0; lostTicks=0; pulsePhase4=0; searchActive=false; searchStep=0; reacquireCount=0;
  mode5HeadingScanActive=false; mode5HeadingScanAttempted=false;
  Serial.print(F("MODE6_ALIGN120_START angle=")); Serial.println(angleDeg);
}

// Mode 5 only: rotation-only reacquisition after consecutive LOST frames.
// No up-front scan is run when the current sensor pair is valid.
void startMode5HeadingScan(unsigned long now) {
  stopAll(); resetMotion();
  mode5HeadingScanActive=true;
  mode5HeadingScanAttempted=true; // never loop heading scans after failure
  mode5HeadingScanStep=0;
  mode5HeadingScanDwell=0;
  mode5HeadingScanStable=0;
  mode5HeadingScanStartMs=now;
  lostTicks=0;
  fourStable=0;
  mode5HeadingPrevValid=false;
  Serial.println(F("ROTATION_SEARCH_BEGIN"));
}

void runMode5HeadingScan(unsigned long now,float theta_est,float rpm1,float rpm2,float rpm3,
                         long c1,long c2,long c3) {
  if (now-mode5HeadingScanStartMs>=MODE5_HEADING_SCAN_TIMEOUT_MS) {
    stopAll();
    mode5HeadingScanActive=false;
    failFour("HEADING_SEARCH_TIMEOUT");
    finishControlTick(c1,c2,c3,now);
    return;
  }
  if (mode5HeadingScanStep>=MODE5_HEADING_SCAN_STEP_COUNT) {
    stopAll();
    mode5HeadingScanActive=false;
    failFour("HEADING_SEARCH_NOT_FOUND");
    finishControlTick(c1,c2,c3,now);
    return;
  }

  // Check sensors EVERY tick, even while rotating between scan waypoints.
  // Unlike the old heading search, do not demand that the ball is already
  // centered: the regular ALIGN 120 controller will correct heading first,
  // then advance when both MID sensors agree sufficiently.
  FourSensors f = readFourSensors();
  bool pairSeen = centralPairValid(f) &&
                  f.midL >= MODE5_HEADING_SCAN_MIN_RAW &&
                  f.midR >= MODE5_HEADING_SCAN_MIN_RAW;
  if (pairSeen && mode5HeadingPrevValid) {
    // Reject an isolated gross ADC jump; real readings may vary while turning.
    pairSeen = (abs(f.midL-mode5HeadingPrevL) <= MODE5_HEADING_SCAN_SAMPLE_JUMP_RAW &&
                abs(f.midR-mode5HeadingPrevR) <= MODE5_HEADING_SCAN_SAMPLE_JUMP_RAW);
  }
  if (pairSeen) {
    if (mode5HeadingScanStable < MODE5_HEADING_SCAN_VALID_TICKS)
      ++mode5HeadingScanStable;
  } else {
    mode5HeadingScanStable=0;
  }
  mode5HeadingPrevL=f.midL;
  mode5HeadingPrevR=f.midR;
  mode5HeadingPrevValid=pairSeen;

  if (mode5HeadingScanStable >= MODE5_HEADING_SCAN_VALID_TICKS) {
    stopAll();
    resetMotion();
    mode5HeadingScanActive=false;
    mode5RotCorrectionActive=false;
    mode5RotStartCount=mode5RotStopCount=0;
    mode5DiffFilterInitialized=false;
    lostTicks=0;
    fourStable=0;
    phaseStartMs=now;
    Serial.print(F("ROTATION_SEARCH_FOUND angle="));
    Serial.println(theta_est*180.0f/PI,1);
    // On next tick regular ALIGN 120 handles direction and distance.
    // It will not move forward until its two-sensor centering gate is met.
    finishControlTick(c1,c2,c3,now);
    return;
  }

  float targetDeg=MODE5_HEADING_SCAN_TARGETS_DEG[mode5HeadingScanStep];
  float errorDeg=targetDeg-theta_est*180.0f/PI;
  if (abs(errorDeg)>MODE5_HEADING_SCAN_TOL_DEG) {
    mode5HeadingScanDwell=0;
    float w=constrain(errorDeg*PI/180.0f*0.95f,
                      -MODE5_HEADING_SCAN_OMEGA_MAX_RAD_S,MODE5_HEADING_SCAN_OMEGA_MAX_RAD_S);
    if (abs(w)<0.12f) w=(w>0?0.12f:-0.12f);
    motionFineOverride=(abs(errorDeg)<=8.0f);
    driveRobot(0.0f,0.0f,w,rpm1,rpm2,rpm3);
  } else {
    stopAll();
    // Only move to another scan angle after a short dwell.
    if (++mode5HeadingScanDwell >= MODE5_HEADING_SCAN_DWELL_TICKS+5) {
      ++mode5HeadingScanStep;
      mode5HeadingScanDwell=0;
      mode5HeadingScanStable=0;
      mode5HeadingPrevValid=false;
      Serial.print(F("ROTATION_SEARCH_STEP="));
      Serial.println(mode5HeadingScanStep);
    }
  }
  finishControlTick(c1,c2,c3,now);
}

bool calculateAlign4(float distance,float &vx,float &vy,bool &valid) {
  FourSensors f=readFourSensors();
  if (mode == ALIGN4) {
    const float raw[4]={f.midL,f.midR,f.nearL,f.nearR};
    printMode5Stability(raw);
  }
  float targetL=(distance==120.0f) ? TARGET_120_MID_L :
                (distance==160.0f) ? TARGET_160_MID_L : TARGET_200_MID_L;
  float targetR=(distance==120.0f) ? TARGET_120_MID_R :
                (distance==160.0f) ? TARGET_160_MID_R : TARGET_200_MID_R;
  float targetAvg=(targetL+targetR)*0.5f;
  float targetDiff=targetL-targetR;
  bool pair=centralPairValid(f);
  bool leftOnly=(f.midL >= MID_SIDE_MIN_RAW && f.midR < MID_CENTER_MIN_RAW);
  bool rightOnly=(f.midR >= MID_SIDE_MIN_RAW && f.midL < MID_CENTER_MIN_RAW);
  // Pair signals can be weak at longer distances; if only one is above the
  // side threshold, use lateral correction without interpreting the average.
  // A strong asymmetry with both sides visible also requests lateral only.
  valid=pair || leftOnly || rightOnly;
  vx=0;vy=0;
  float eDiff=targetDiff-f.diff;
  mode5LastPair = pair;
  mode5LastAvg = f.avg;
  mode5LastDiffError = eDiff;
  Serial.print(F("ALIGN4 ")); Serial.print(distance,0);
  Serial.print(F(" MID="));Serial.print(f.midL,0);Serial.print(',');Serial.print(f.midR,0);
  Serial.print(F(" NEAR="));Serial.print(f.nearL,0);Serial.print(',');Serial.print(f.nearR,0);
  Serial.print(F(" D="));Serial.print(f.diff,0);
  Serial.print(F(" state="));
  if(pair) Serial.println(F("PAIR"));
  else if(leftOnly) Serial.println(F("LEFT_ONLY"));
  else if(rightOnly) Serial.println(F("RIGHT_ONLY"));
  else Serial.println(F("LOST"));
  if (!valid) { fourStable=0; return false; }

  // Use only lateral correction when a single mid sensor sees the ball.
  // No distance inference from one sensor; no near-only autonomous chase.
  if (leftOnly || rightOnly || abs(eDiff) > ALIGN_CENTER_DIFF_GATE) {
    fourStable=0;
    vy=constrain(SENSOR_K_DIFF_RAW*f.diff - SENSOR_K_DIFF_RAW*targetDiff,
                 -SENSOR_V_LIMIT_DIFF,SENSOR_V_LIMIT_DIFF);
    if(abs(vy)<SENSOR_V_MIN_DIFF)
      vy=(vy>=0)?SENSOR_V_MIN_DIFF:-SENSOR_V_MIN_DIFF;
    return false;
  }

  // Centered and both sides detected: only now average is usable for distance.
  float eAvg=targetAvg-f.avg;
  if (abs(eAvg)<=ALIGN_AVG_TOL_RAW_4 && abs(eDiff)<=ALIGN_DIFF_TOL_RAW_4) {
    if (fourStable<ALIGN_STABLE_READS) ++fourStable;
    return (fourStable>=ALIGN_STABLE_READS);
  }
  fourStable=0;
  if (abs(eDiff)>ALIGN_DIFF_TOL_RAW_4) {
    vy=constrain(-SENSOR_K_DIFF_RAW*eDiff,-SENSOR_V_LIMIT_DIFF,SENSOR_V_LIMIT_DIFF);
    if (abs(vy)<SENSOR_V_MIN_DIFF) vy=(vy>=0)?SENSOR_V_MIN_DIFF:-SENSOR_V_MIN_DIFF;
  } else {
    // Long-range straight approach: continuous encoder-feedback motor drive.
    // Switch to fine pulses automatically near the selected 120/160/200mm target.
    alignCoarseDrive = (eAvg > 55.0f);
    if (alignCoarseDrive) vx = 65.0f;
    else {
      vx=constrain(SENSOR_K_AVG_RAW*eAvg,-SENSOR_V_LIMIT_AVG,SENSOR_V_LIMIT_AVG);
      if (abs(vx)<SENSOR_V_MIN_AVG) vx=(vx>=0)?SENSOR_V_MIN_AVG:-SENSOR_V_MIN_AVG;
    }
  }
  return false;
}

// =====================
// Encoder ISRs
// =====================
void readEncoderM1() {
  int a = digitalRead(M1_ENC_A);
  int b = digitalRead(M1_ENC_B);

  if (a == b) countM1++;
  else countM1--;
}

void readEncoderM2() {
  int a = digitalRead(M2_ENC_A);
  int b = digitalRead(M2_ENC_B);

  if (a == b) countM2++;
  else countM2--;
}

void readEncoderM3() {
  int a = digitalRead(M3_ENC_A);
  int b = digitalRead(M3_ENC_B);

  if (a == b) countM3++;
  else countM3--;
}
