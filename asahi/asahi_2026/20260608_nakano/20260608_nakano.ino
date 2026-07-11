// 3-wheel omni robot + last-year style 10-digit command mode
//
// Last-year numeric command format (10 digits):
// [0]   mode
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
// 2: HIT_RESERVED         reserved for future servo hitting mechanism
// 3: SENSOR_RESERVED      reserved for future sensor/recognition mode
// 4: AUTO_ABSOLUTE        encoder-based rotate then move, similar role to last year mode 4
// 5: ALIGN_0_8            same as text command: ALIGN 0.8
// 6: BALLROT_0_8          ALIGN 0.8 -> small BALLROT step -> ALIGN 0.8 repeatedly
//
// Text commands are also left for bench testing:
// X 100 / Y 100 / ROT 30 / ARC 80 0 30 / SENSOR / ALIGN 1.5 / BALLROT 30 1.5 / STOP

#include <Arduino.h>
#include <math.h>

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
const int IR_SAMPLE_COUNT = 20;

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

const float L_WHEEL_MM = 140.0;

// =====================
// Control constants
// =====================
const float M1_GAIN = 1.00;
const float M2_GAIN = 1.00;
const float M3_GAIN = 1.00;

const float KP_RPM = 0.5;

const float M1_PWM_GAIN = 0.50;
const float M2_PWM_GAIN = 0.50;
const float M3_PWM_GAIN = 0.50;

const float PWM_STEP_LIMIT = 1.0;  // reserved. Not used yet.
const int DEADZONE_PWM = 40;
const int PWM_LIMIT = 255;

const unsigned long CONTROL_INTERVAL = 100;

// motion speed for encoder-based commands
const float MOVE_SPEED_MM_S = 220.0;
const float MOVE_SLOW_MM_S = 60.0;
const float ROT_SPEED_RAD_S = 0.35;
const float ARC_SPEED_RAD_S = 0.25;

// stepwise BALLROT setting
const float BALLROT_STEP_DEG = 5.0;      // rotate this many degrees, then ALIGN again
const float BALLROT_RADIUS_MM = 80.0;    // same virtual center offset as the old BALLROT

// motion speed for vision-style mode 1
const float VISION_MOVE_FAST_MM_S = 180.0;
const float VISION_MOVE_SLOW_MM_S = 60.0;
const float VISION_ROT_FAST_RAD_S = 0.35;
const float VISION_ROT_SLOW_RAD_S = 0.15;
const float VISION_DIST_SLOW_BAND_MM = 350.0;
const float VISION_ANGLE_SLOW_BAND_DEG = 8.0;

// stop tolerance
const float MOVE_TOL_MM = 5.0;
const float ROT_TOL_RAD = 3.0 * PI / 180.0;
const float VISION_DIST_TOL_MM = 10.0;
const float VISION_ANGLE_TOL_DEG = 2.0;

// sensor control
const float SENSOR_TOL_V = 0.03;
const float SENSOR_K_AVG = 90.0;    // front-back correction mm/s per V
const float SENSOR_K_DIFF = 70.0;   // left-right correction mm/s per V
const float SENSOR_V_LIMIT = 100.0; // mm/s

const float SENSOR_DETECT_V = 0.20;
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
  HIT_RESERVED,      // mode 2: future servo mechanism. Empty on purpose.
  SENSOR_RESERVED,   // mode 3: future detection mode. Empty on purpose.
  AUTO_ABSOLUTE      // mode 4: rotate then move using encoders
};

Mode mode = IDLE;

float targetValue = 0.0;
float dirSign = 1.0;

float arcCx = 80.0;
float arcCy = 0.0;

float targetSensorV = 1.5;

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
float ballRotTargetSensorV = 0.8;

// Numeric one-shot/restart guard
int lastStartedNumericMode = -1;
int lastStartedAngleDeg = 0;
int lastStartedDistMm = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

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
  Serial.println(F("Mode 0 STOP / 1 VISION_MOVE / 2 HIT_RESERVED / 3 SENSOR_RESERVED / 4 AUTO_ABSOLUTE / 5 ALIGN_0.8 / 6 BALLROT_0.8"));
  Serial.println(F("Text test: X 100 / Y 100 / ROT 30 / ARC 80 0 30 / SENSOR / ALIGN 1.5 / BALLROT 30 1.5 / STOP"));
}

void loop() {
  readLatestCommand();

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
  int angleSigned = (signFlagDeg == 1) ? angleAbs : -angleAbs;

  int signFlagDis = line.substring(5, 6).toInt();
  int disAbs = line.substring(6, 10).toInt();
  int disSigned = (signFlagDis == 1) ? disAbs : -disAbs;

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
      if (!isSameNumericCommand(modeVal, angleSigned, disSigned) || mode != HIT_RESERVED) {
        startHitReserved(angleSigned, disSigned);
        rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      }
      break;

    case 3:
      if (!isSameNumericCommand(modeVal, angleSigned, disSigned) || mode != SENSOR_RESERVED) {
        startSensorReserved(angleSigned, disSigned);
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
      // Same as text command: ALIGN 0.8
      // Do not restart the search/alignment while the same mode is continuously sent.
      targetSensorV = 0.8;
      if (!(mode == ALIGN_SEARCH_LEFT || mode == ALIGN_SEARCH_RIGHT || mode == ALIGN_BALL) ||
          !isSameNumericCommand(modeVal, angleSigned, disSigned)) {
        startAlign(0.8);
        rememberStartedNumericCommand(modeVal, angleSigned, disSigned);
      }
      break;

    case 6:
      // Stepwise BALLROT:
      // ALIGN 0.8 first. If the ball is not found, finish here.
      // Then rotate by a small angle, ALIGN again, and repeat.
      // The angle comes from the numeric command. Distance digits are ignored.
      targetSensorV = 0.8;
      if (!isBallRotStepwiseMode() || !isSameNumericCommand(modeVal, angleSigned, disSigned)) {
        startBallRot((float)angleSigned, 0.8);
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

  if (line == "SENSOR") {
    printSensor();
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
    float v = getArg(line, 1);
    if (v <= 0.0) v = 1.5;
    startAlign(v);
  }
  else if (cmd == "BALLROT") {
    float deg = getArg(line, 1);
    float v = getArg(line, 2);
    if (v <= 0.0) v = 1.5;
    startBallRot(deg, v);
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

void startAlign(float targetV) {
  resetMotion();

  targetSensorV = targetV;

  float vL, vR, vAvg, vDiff;
  readSensorVoltages(vL, vR, vAvg, vDiff);

  if (vAvg <= SENSOR_DETECT_V) {
    mode = ALIGN_SEARCH_LEFT;
    targetValue = ALIGN_SEARCH_RANGE_MM;
    dirSign = 1.0;

    Serial.println(F("ALIGN: no object detected. Search left."));
  } else {
    mode = ALIGN_BALL;
    Serial.print(F("Start ALIGN targetV="));
    Serial.println(targetV, 3);
  }
}

void startBallRot(float deg, float targetV) {
  // Stepwise BALLROT:
  // 1) ALIGN targetV. If the ball is not detected during search, finish.
  // 2) Rotate a small angle.
  // 3) ALIGN again.
  // 4) Repeat until the requested total angle is reached.
  resetMotion();

  targetSensorV = targetV;
  ballRotTargetSensorV = targetV;
  ballRotTotalRad = abs(deg) * PI / 180.0;
  ballRotDoneRad = 0.0;
  ballRotDirSign = (deg >= 0) ? 1.0 : -1.0;

  float vL, vR, vAvg, vDiff;
  readSensorVoltages(vL, vR, vAvg, vDiff);

  if (vAvg <= SENSOR_DETECT_V) {
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
    Serial.print(F(", targetV="));
    Serial.print(targetV, 3);
    Serial.println(F(". First ALIGN."));
  }
}

void startHitReserved(float angleDeg, float distanceMm) {
  // Mode 2 is intentionally kept as an empty slot for the future servo hitting mechanism.
  // Add servo attach/write sequence here later.
  stopAll();
  mode = HIT_RESERVED;

  Serial.print(F("Mode 2 HIT_RESERVED. angle="));
  Serial.print(angleDeg);
  Serial.print(F(" dist="));
  Serial.println(distanceMm);
}

void startSensorReserved(float angleDeg, float distanceMm) {
  // Mode 3 is intentionally kept as an empty slot for a future sensor/search mode.
  // If needed, current IR alignment can be moved here later.
  stopAll();
  mode = SENSOR_RESERVED;

  Serial.print(F("Mode 3 SENSOR_RESERVED. angle="));
  Serial.print(angleDeg);
  Serial.print(F(" dist="));
  Serial.println(distanceMm);
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

  float x_est = (s3 - s1) / 1.732;
  float y_est = (s1 + s3 - 2.0 * s2) / 3.0;
  float theta_est = (s1 + s2 + s3) / (3.0 * L_WHEEL_MM);

  if (mode == IDLE || mode == HIT_RESERVED || mode == SENSOR_RESERVED) {
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

  if (mode == VISION_MOVE) {
    computeVisionMoveCommand(Vx, Vy, omega);
    done = false;  // Keep this mode alive until mode 0 or another command arrives.
  }
  else if (mode == MOVE_X) {
    float progress = dirSign * x_est;
    float remain = targetValue - progress;

    if (remain <= MOVE_TOL_MM) {
      done = true;
    } else {
      float speed = (remain < 35.0) ? MOVE_SLOW_MM_S : MOVE_SPEED_MM_S;
      Vx = speed * dirSign;
    }
  }
  else if (mode == MOVE_Y) {
    float progress = dirSign * y_est;
    float remain = targetValue - progress;

    if (remain <= MOVE_TOL_MM) {
      done = true;
    } else {
      float speed = (remain < 35.0) ? MOVE_SLOW_MM_S : MOVE_SPEED_MM_S;
      Vy = speed * dirSign;
    }
  }
  else if (mode == ROT_CENTER) {
    float progress = dirSign * theta_est;
    float remain = targetValue - progress;

    if (remain <= ROT_TOL_RAD) {
      done = true;
    } else {
      omega = (remain < 0.25) ? 0.18 * dirSign : ROT_SPEED_RAD_S * dirSign;
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
    readSensorVoltages(vL, vR, vAvg, vDiff);

    float progress = y_est;

    if (vAvg > SENSOR_DETECT_V) {
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

    Serial.print(F("SEARCH_LEFT Avg="));
    Serial.print(vAvg, 3);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == ALIGN_SEARCH_RIGHT) {
    float vL, vR, vAvg, vDiff;
    readSensorVoltages(vL, vR, vAvg, vDiff);

    float progress = -y_est;

    if (vAvg > SENSOR_DETECT_V) {
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

    Serial.print(F("SEARCH_RIGHT Avg="));
    Serial.print(vAvg, 3);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == ALIGN_BALL) {
    float vL, vR, vAvg, vDiff;
    readSensorVoltages(vL, vR, vAvg, vDiff);

    float errorAvg = targetSensorV - vAvg;
    float errorDiff = 0.0 - vDiff;

    const float DIFF_PRIORITY_V = 0.07;

    if (abs(errorAvg) < SENSOR_TOL_V && abs(errorDiff) < SENSOR_TOL_V) {
      done = true;
    }
    else {
      if (abs(errorDiff) > DIFF_PRIORITY_V) {
        Vx = 0.0;
        Vy = -constrain(SENSOR_K_DIFF * errorDiff,
                        -SENSOR_V_LIMIT,
                         SENSOR_V_LIMIT);
      }
      else {
        Vy = 0.0;
        Vx = constrain(SENSOR_K_AVG * errorAvg,
                       -SENSOR_V_LIMIT,
                        SENSOR_V_LIMIT);
      }

      omega = 0.0;
    }

    Serial.print(F("ALIGN | L="));
    Serial.print(vL, 3);
    Serial.print(F(" R="));
    Serial.print(vR, 3);
    Serial.print(F(" Avg="));
    Serial.print(vAvg, 3);
    Serial.print(F(" Diff="));
    Serial.print(vDiff, 3);
    Serial.print(F(" Vx="));
    Serial.print(Vx);
    Serial.print(F(" Vy="));
    Serial.println(Vy);
  }
  else if (mode == BALL_ROT_SEARCH_LEFT) {
    float vL, vR, vAvg, vDiff;
    readSensorVoltages(vL, vR, vAvg, vDiff);

    float progress = y_est;

    if (vAvg > SENSOR_DETECT_V) {
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

    Serial.print(F("BALLROT_SEARCH_LEFT Avg="));
    Serial.print(vAvg, 3);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == BALL_ROT_SEARCH_RIGHT) {
    float vL, vR, vAvg, vDiff;
    readSensorVoltages(vL, vR, vAvg, vDiff);

    float progress = -y_est;

    if (vAvg > SENSOR_DETECT_V) {
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

    Serial.print(F("BALLROT_SEARCH_RIGHT Avg="));
    Serial.print(vAvg, 3);
    Serial.print(F(" y="));
    Serial.println(y_est);
  }
  else if (mode == BALL_ROT_ALIGN) {
    float vL, vR, vAvg, vDiff;
    readSensorVoltages(vL, vR, vAvg, vDiff);

    float errorAvg = ballRotTargetSensorV - vAvg;
    float errorDiff = 0.0 - vDiff;

    const float DIFF_PRIORITY_V = 0.07;

    bool aligned = (abs(errorAvg) < SENSOR_TOL_V && abs(errorDiff) < SENSOR_TOL_V);

    if (aligned) {
      float remainingTotal = ballRotTotalRad - ballRotDoneRad;

      if (remainingTotal <= ROT_TOL_RAD) {
        done = true;
      } else {
        float stepRad = BALLROT_STEP_DEG * PI / 180.0;
        if (stepRad > remainingTotal) stepRad = remainingTotal;

        resetMotion();
        mode = BALL_ROT_STEP;
        targetValue = stepRad;
        dirSign = ballRotDirSign;

        Serial.print(F("BALLROT: ALIGN done. Start small step deg="));
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
    else {
      if (abs(errorDiff) > DIFF_PRIORITY_V) {
        Vx = 0.0;
        Vy = -constrain(SENSOR_K_DIFF * errorDiff,
                        -SENSOR_V_LIMIT,
                         SENSOR_V_LIMIT);
      }
      else {
        Vy = 0.0;
        Vx = constrain(SENSOR_K_AVG * errorAvg,
                       -SENSOR_V_LIMIT,
                        SENSOR_V_LIMIT);
      }

      omega = 0.0;
    }

    Serial.print(F("BALLROT_ALIGN | L="));
    Serial.print(vL, 3);
    Serial.print(F(" R="));
    Serial.print(vR, 3);
    Serial.print(F(" Avg="));
    Serial.print(vAvg, 3);
    Serial.print(F(" Diff="));
    Serial.print(vDiff, 3);
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
        done = true;
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
      readSensorVoltages(vL, vR, vAvg, vDiff);

      float errorAvg = ballRotTargetSensorV - vAvg;
      float errorDiff = 0.0 - vDiff;

      omega = (remain < 0.08) ? 0.15 * dirSign : ARC_SPEED_RAD_S * dirSign;

      Vx = constrain(SENSOR_K_AVG * errorAvg, -SENSOR_V_LIMIT, SENSOR_V_LIMIT);
      Vy = -constrain(SENSOR_K_DIFF * errorDiff, -SENSOR_V_LIMIT, SENSOR_V_LIMIT);

      // Rotation around a virtual point near the ball.
      Vx += omega * 0.0;
      Vy += -omega * BALLROT_RADIUS_MM;

      Serial.print(F("BALLROT_STEP IR Avg="));
      Serial.print(vAvg, 3);
      Serial.print(F(", Diff="));
      Serial.print(vDiff, 3);
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
        float speed = (remain < 35.0) ? MOVE_SLOW_MM_S : MOVE_SPEED_MM_S;
        Vx = speed * dirSign;
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
    float w = (abs(eDeg) > VISION_ANGLE_SLOW_BAND_DEG) ? VISION_ROT_FAST_RAD_S : VISION_ROT_SLOW_RAD_S;
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

  updateMotor(targetRpm1, rpm1, pwmM1, M1_PWM_GAIN);
  updateMotor(targetRpm2, rpm2, pwmM2, M2_PWM_GAIN);
  updateMotor(targetRpm3, rpm3, pwmM3, M3_PWM_GAIN);

  setMotor(M1_IN1, M1_IN2, (int)pwmM1);
  setMotor(M2_IN1, M2_IN2, (int)pwmM2);
  setMotor(M3_IN1, M3_IN2, (int)pwmM3);
}

void updateMotor(float targetRpm, float currentRpm, float &pwm, float gain) {
  if (abs(targetRpm) < 0.5) {
    pwm = 0;
    return;
  }

  float basePwm = rpmToBasePwm(targetRpm) * gain;

  float error = targetRpm - currentRpm;
  float correction = KP_RPM * error;

  pwm = basePwm + correction;

  pwm = constrain(pwm, -PWM_LIMIT, PWM_LIMIT);
}

float rpmToBasePwm(float targetRpm) {
  float r = abs(targetRpm);

  if (r < 1.0) return 0;

  // Simple measured-value approximation.
  float base = 100.0 + 2.0 * r;

  if (base > PWM_LIMIT) base = PWM_LIMIT;

  if (targetRpm > 0) {
    return base;
  } else {
    return -base;
  }
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

  analogWrite(M1_IN1, 255);
  analogWrite(M1_IN2, 255);

  analogWrite(M2_IN1, 255);
  analogWrite(M2_IN2, 255);

  analogWrite(M3_IN1, 255);
  analogWrite(M3_IN2, 255);
}

// =====================
// Sensor
// =====================
int readAverageRaw(int pin) {
  long sum = 0;

  for (int i = 0; i < IR_SAMPLE_COUNT; i++) {
    sum += analogRead(pin);
    delay(2);
  }

  return sum / IR_SAMPLE_COUNT;
}

void readSensorVoltages(float &vL, float &vR, float &vAvg, float &vDiff) {
  int rawL = readAverageRaw(IR_L_PIN);
  int rawR = readAverageRaw(IR_R_PIN);

  vL = rawL * (5.0 / 1023.0);
  vR = rawR * (5.0 / 1023.0);

  vDiff = vL - vR;
  vAvg = (vL + vR) / 2.0;
}

void printSensor() {
  float vL, vR, vAvg, vDiff;
  readSensorVoltages(vL, vR, vAvg, vDiff);

  Serial.print(F("L="));
  Serial.print(vL, 3);
  Serial.print(F(" V, R="));
  Serial.print(vR, 3);
  Serial.print(F(" V, Diff="));
  Serial.print(vDiff, 3);
  Serial.print(F(" V, Avg="));
  Serial.print(vAvg, 3);
  Serial.println(F(" V"));
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
