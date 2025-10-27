#include <Arduino.h>
#include <math.h>
#include <Servo.h>
#include "nakano.h"

#define DEBUG

//============== SERVO: takemichi ==================//
Servo servoA;
Servo servoB;
Servo servoC;
int angleA = 180;
int angleB = 40;
int angleC = 105;

//============== MOVE: Nakano ==================//
// 長いためヘッダーファイルに分割

// その他
const int EXPECTED_LEN = 10;

void setup() {
  // シリアル開始（※コメントと実値を一致させています）
  Serial.begin(115200);
  Serial.setTimeout(50);  // readStringUntilの待ち時間（ms）

  // takemichi servo setup
  servoA.attach(9);   // デカサーボ
  servoB.attach(10);  // ミニサーボ
  servoC.attach(11);  // 角度サーボ

  servoA.write(angleA);
  servoB.write(angleB);
  servoC.write(angleC);

  // nakano setup
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
  t_next = micros();
}
void loop() {

  // ---------------------------------
  // 1. シリアル受信
  // ---------------------------------
  if (Serial.available()) {
    String receivedData = Serial.readStringUntil('\n');
    receivedData.trim();
#ifdef DEBUG
    Serial.print("Received Data: ");
    Serial.println(receivedData);
#endif

    if (receivedData.length() == EXPECTED_LEN) {
      // 全桁チェック
      bool all_digit = true;
      for (unsigned int i = 0; i < receivedData.length(); ++i) {
        if (!isDigit(receivedData.charAt(i))) {
          all_digit = false;
          break;
        }
      }
      if (all_digit) {
        // パース
        int mode = receivedData.substring(0, 1).toInt();

        int sign_flag_deg = receivedData.substring(1, 2).toInt();
        int angle_abs     = receivedData.substring(2, 5).toInt();
        int angle_signed  = (sign_flag_deg == 1) ? angle_abs : -angle_abs;

        int sign_flag_dis = receivedData.substring(5, 6).toInt();
        int dis_abs       = receivedData.substring(6, 10).toInt();
        int dis_signed    = (sign_flag_dis == 1) ? dis_abs : -dis_abs;

        mode_val   = mode;
        angle_deg  = angle_signed;
        dis_val_mm = dis_signed;

#ifdef DEBUG
        Serial.print(F("mode_val: "));
        Serial.println(mode_val);
        Serial.print(F("angle_deg: "));
        Serial.println(angle_deg);
        Serial.print(F("dis_val: "));
        Serial.println(dis_val_mm);
#endif
      }
    }
  }

  // ---------------------------------
  // 2. ステッピングは周期制御
  // ---------------------------------
  unsigned long now = micros();
  if ((long)(now - t_next) < 0) {
    return;  // まだ周期じゃないから今回はここで終了
  }
  float dt = (float)UPDATE_PERIOD_US * 1e-6f;  // 0.001
  t_next += UPDATE_PERIOD_US;

  // ---------------------------------
  // 3. メイン動作 
  // ---------------------------------
  switch (mode_val) {
    case 1: {
#ifdef DEBUG
      Serial.println("move mode");
#endif
      float x_cmd = 0.0f, y_cmd = 0.0f;
      computeCommandRates(x_cmd, y_cmd); // 角度優先→距離
      driveVelocity(x_cmd, y_cmd, dt);   // ステップ吐く
      break;
    }

    case 2: {
#ifdef DEBUG
      Serial.println("hitting mode");
#endif
      int val = (257 - dis_val_mm) / 2;
      angleA = val;

      // ひとまずそのままやるならこう
      delay(1000);
      servoB.write(150);
      delay(1000);
      servoA.write(angleA);
      delay(2000);
      servoB.write(40);
      delay(2000);
      servoA.write(180);
      break;
    }

    default: {
#ifdef DEBUG
      Serial.println("Non mode");
#endif
      // mode_valが1でも2でもない → 何もしない（停止）
      break;
    }
  }
}
