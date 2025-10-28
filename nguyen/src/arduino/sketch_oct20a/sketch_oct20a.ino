#include <Arduino.h>
#include <math.h>

// define
// #define DEBUG

// ===== CNC Shield V3 (UNO) pins (GRBL準拠) =====
const int X_STEP = 2;
const int X_DIR = 5;
const int Y_STEP = 3;
const int Y_DIR = 6;
const int EN_PIN = 8;  // A4988: LOW=有効, HIGH=無効

// ===== Motor / driver config =====
const int FULL_STEPS_PER_REV = 200;  // 1.8°/step
const int MICROSTEP = 8;             // ← 実機ジャンパに合わせる（1,2,4,8,16）
const long STEPS_PER_REV = (long)FULL_STEPS_PER_REV * MICROSTEP;

const unsigned int STEP_PULSE_US = 3;         // STEPパルス幅（2〜5us）
const unsigned long UPDATE_PERIOD_US = 1000;  // 制御周期（1kHz）

// ===== 進行方向の符号 =====
const int FORWARD_X_SIGN = +1;
const int FORWARD_Y_SIGN = -1;
const int ROT_CCW_X_SIGN = +1;
const int ROT_CCW_Y_SIGN = -1;

// ===== 速度・調整 =====
const float LIN_FAST_RPS = 1.00f;      // 直進：遠い時
const float LIN_SLOW_RPS = 0.30f;      // 直進：近傍
const float ROT_FAST_RPS = 0.50f;      // 回頭：遠い時
const float ROT_SLOW_RPS = 0.10f;      // 回頭：近傍
const float ROT_CORR_RPS_MAX = 0.60f;  // 直進中の角度補正上限（今回は最小限利用）

// 直進中の角度補正ゲイン（微調整用。不要なら0でもOK）
const float K_ROT_CORR = 0.0f;  // [rev/s per deg]

// ===== 許容・減速帯 =====
const float ANGLE_TOL_DEG = 1.0f;        // 角度完了帯
const float ANGLE_SLOW_BAND_DEG = 5.0f;  // 角度減速帯
const float DIST_TOL_MM = 1.0f;          // 距離完了帯
const float DIST_SLOW_BAND_MM = 500.0f;  // 距離減速帯

// ===== ハード上限 =====
const float HARD_MAX_RPS = 3.0f;  // ドライバ/機械の上限ガード

// ===== DIR極性（必要なら反転）=====
const bool X_DIR_POS = true;  // true: +方向でHIGH、false: +方向でLOW
const bool Y_DIR_POS = true;

// ========== 内部状態 ==========
float x_step_accum = 0.0f;
float y_step_accum = 0.0f;
int x_dir_sign = +1;
int y_dir_sign = +1;

// 受信解析後の最新コマンド
volatile int mode_val = 0;    // 1で動作、他は停止
volatile int angle_deg = 0;   // −180..+180 の角度誤差（＋=CCW）
volatile int dis_val_mm = 0;  // 距離誤差[mm]（＋=前進、−=後退）

unsigned long t_next = 0;

// ----------------- ユーティリティ -----------------
inline void pulseStep(int pin) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(STEP_PULSE_US);
  digitalWrite(pin, LOW);
  delayMicroseconds(STEP_PULSE_US);
}

inline void setDirBySign(int dirPin, bool dirPos, int sign) {
  bool level = (sign > 0) ? (dirPos ? HIGH : LOW) : (dirPos ? LOW : HIGH);
  digitalWrite(dirPin, level);
}

inline float clampAbs(float v, float vmax) {
  if (v > vmax) return vmax;
  if (v < -vmax) return -vmax;
  return v;
}

// rev/s指令→STEP出力（非ブロッキング）
void driveVelocity(float x_rev_s, float y_rev_s, float dt_s) {
  x_rev_s = clampAbs(x_rev_s, HARD_MAX_RPS);
  y_rev_s = clampAbs(y_rev_s, HARD_MAX_RPS);

  int x_sign = (x_rev_s >= 0.0f) ? +1 : -1;
  int y_sign = (y_rev_s >= 0.0f) ? +1 : -1;
  if (x_sign != x_dir_sign) {
    setDirBySign(X_DIR, X_DIR_POS, x_sign);
    x_dir_sign = x_sign;
  }
  if (y_sign != y_dir_sign) {
    setDirBySign(Y_DIR, Y_DIR_POS, y_sign);
    y_dir_sign = y_sign;
  }

  float x_steps_add = fabsf(x_rev_s) * (float)STEPS_PER_REV * dt_s;
  float y_steps_add = fabsf(y_rev_s) * (float)STEPS_PER_REV * dt_s;
  x_step_accum += x_steps_add;
  y_step_accum += y_steps_add;

  while (x_step_accum >= 1.0f) {
    pulseStep(X_STEP);
    x_step_accum -= 1.0f;
  }
  while (y_step_accum >= 1.0f) {
    pulseStep(Y_STEP);
    y_step_accum -= 1.0f;
  }
}

// ===== シリアル受信（改行区切り10桁） =====
// 形式: [0]=mode, [1]=sign_deg, [2-4]=angle_abs, [5]=sign_dis, [6-9]=dis_abs, "\n"
void readLineAndParseIfAny() {
  if (Serial.available() <= 0) return;

  String receivedData = Serial.readStringUntil('\n');  // setTimeoutに従って待つ
  receivedData.trim();                                 // CR/LF/空白除去

  if (receivedData.length() == 0) return;

  const int EXPECTED_LEN = 10;
  Serial.print(F("Received Data: "));
  Serial.println(receivedData);

  if ((int)receivedData.length() != EXPECTED_LEN) {
#ifdef DEBUG
    Serial.print(F("Error: Unexpected length (expected "));
    Serial.print(EXPECTED_LEN);
    Serial.print(F(") got "));
    Serial.println(receivedData.length());
#endif
    return;
  }

  // 全桁が数字か確認
  for (int i = 0; i < EXPECTED_LEN; ++i) {
    if (!isDigit(receivedData.charAt(i))) {
      Serial.println(F("Error: non-digit found"));
      return;
    }
  }

  int mode = receivedData.substring(0, 1).toInt();

  int sign_flag_deg = receivedData.substring(1, 2).toInt();  // 0 or 1
  if (sign_flag_deg != 0 && sign_flag_deg != 1) {
    Serial.println(F("Error: invalid sign flag (deg)"));
    return;
  }
  int angle_abs = receivedData.substring(2, 5).toInt();  // 000..999
  int angle_signed = (sign_flag_deg == 1) ? angle_abs : -angle_abs;

  int sign_flag_dis = receivedData.substring(5, 6).toInt();  // 0 or 1
  if (sign_flag_dis != 0 && sign_flag_dis != 1) {
    Serial.println(F("Error: invalid sign flag (dis)"));
    return;
  }
  int dis_abs = receivedData.substring(6, 10).toInt();  // 0..9999
  int dis_signed = (sign_flag_dis == 1) ? dis_abs : -dis_abs;

  // 共有状態を更新
  mode_val = mode;
  angle_deg = angle_signed;  // −ならCW, ＋ならCCWに回すべき誤差
  dis_val_mm = dis_signed;   // −なら後退、＋なら前進

  Serial.print(F("mode_val: "));
  Serial.println(mode_val);
  Serial.print(F("angle_deg: "));
  Serial.println(angle_deg);
  Serial.print(F("dis_val: "));
  Serial.println(dis_val_mm);
}

// ===== 上位ロジック：mode/angle/dis → x,y rev/s =====
void computeCommandRates(float& x_cmd, float& y_cmd) {
  x_cmd = 0.0f;
  y_cmd = 0.0f;

  // 1) 角度合わせ（優先）
  int e_deg = angle_deg;  // すでに −/＋ で誤差。＋=CCW
  if (abs(e_deg) > ANGLE_TOL_DEG) {
    float w_rev_s = (abs(e_deg) > ANGLE_SLOW_BAND_DEG) ? ROT_FAST_RPS : ROT_SLOW_RPS;
    w_rev_s *= (e_deg >= 0) ? +1.0f : -1.0f;

    // その場回頭（左右輪逆転）
    x_cmd = ROT_CCW_X_SIGN * w_rev_s;
    y_cmd = ROT_CCW_Y_SIGN * (-w_rev_s);
    return;
  }

  // 2) 距離制御（符号付き：＋=前進、−=後退）
  int d_mm = dis_val_mm;
  if (abs(d_mm) > DIST_TOL_MM) {
    float v_mag = (abs(d_mm) > DIST_SLOW_BAND_MM) ? LIN_FAST_RPS : LIN_SLOW_RPS;
    float sign = (d_mm >= 0) ? +1.0f : -1.0f;  // ＋前進, −後退

    float lin_x = FORWARD_X_SIGN * (v_mag * sign);
    float lin_y = FORWARD_Y_SIGN * (v_mag * sign);

    // 直進中の方位補正（小さめ）
    float w_corr = clampAbs(K_ROT_CORR * (float)e_deg, ROT_CORR_RPS_MAX);
    float rot_x = ROT_CCW_X_SIGN * w_corr;
    float rot_y = ROT_CCW_Y_SIGN * (-w_corr);

    x_cmd = lin_x + rot_x;
    y_cmd = lin_y + rot_y;
    return;
  }

  // 3) 整定 → 停止
  x_cmd = 0.0f;
  y_cmd = 0.0f;
}

void setup() {
  pinMode(X_STEP, OUTPUT);
  pinMode(X_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Y_DIR, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  digitalWrite(EN_PIN, LOW);  // ドライバ有効化（A4988はLOWで有効）

  setDirBySign(X_DIR, X_DIR_POS, +1);
  setDirBySign(Y_DIR, Y_DIR_POS, +1);
  x_dir_sign = +1;
  y_dir_sign = +1;

  // シリアル開始（※コメントと実値を一致させています）
  Serial.begin(115200);
  Serial.setTimeout(50);  // readStringUntilの待ち時間（ms）

  Serial.println(F("UNO+CNC Shield RT ctrl (10-digit framed by \\n) ready."));
  Serial.println(F("Frame: m s aaa t dddd  -> m=mode(1=run), s=deg sign(0: -,1:+), aaa=|deg|,"));
  Serial.println(F("                                  t=dist sign(0: -,1:+), dddd=|mm|"));
  t_next = micros();
}

void loop() {
  // 1) フレーム受信＆解析（mode_val, angle_deg, dis_val_mm を更新）
  readLineAndParseIfAny();

  // 2) 1kHzの時間ゲート（常にここで回す）
  unsigned long now = micros();
  if ((long)(now - t_next) < 0) return;        // まだ周期に満たないなら何もしない
  float dt = (float)UPDATE_PERIOD_US * 1e-6f;  // 例: 0.001
  t_next += UPDATE_PERIOD_US;                  // 次の実行時刻へ

  // 3) mode判別（mode1のみ動作、他は停止指令）
  float x_cmd = 0.0f, y_cmd = 0.0f;
  if (mode_val == 1) {
    computeCommandRates(x_cmd, y_cmd);  // 角度優先→距離
    driveVelocity(x_cmd, y_cmd, dt);
  }
}
