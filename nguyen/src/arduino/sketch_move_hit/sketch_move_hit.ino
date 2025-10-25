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
  // ======================================================
  // 1. シリアル受信
  // ======================================================
  if (Serial.available()) {
    String receivedData = Serial.readStringUntil('\n');
    receivedData.trim();  // remove CR/LF and spaces
#ifdef DEBUG
    Serial.print("Received Data: ");
    Serial.println(receivedData);
#endif

    // 文字列数確認
    if (receivedData.length() == EXPECTED_LEN) {
      // すべてが数字であるかを確認
      for (unsigned int i = 0; i < receivedData.length(); ++i) {
        if (!isDigit(receivedData.charAt(i)))
          return;
      }
      // 各変数に格納 //
      // mode
      int mode = receivedData.substring(0, 1).toInt();

      // 角度
      int sign_flag_deg = receivedData.substring(1, 2).toInt();  // 0 or 1
      int angle_abs = receivedData.substring(2, 5).toInt();      // 000..999
      int angle_signed = (sign_flag_deg == 1) ? angle_abs : -angle_abs;

      // 距離
      int sign_flag_dis = receivedData.substring(5, 6).toInt();  // 0 or 1
      int dis_abs = receivedData.substring(6, 10).toInt();       // 0..9999
      int dis_signed = (sign_flag_dis == 1) ? dis_abs : -dis_abs;

      // 共有状態を更新
      mode_val = mode;
      angle_deg = angle_signed;  // −ならCW, ＋ならCCWに回すべき誤差
      dis_val_mm = dis_signed;   // −なら後退、＋なら前進
#ifdef DEBUG
      Serial.print(F("mode_val: "));
      Serial.println(mode_val);
      Serial.print(F("angle_deg: "));
      Serial.println(angle_deg);
      Serial.print(F("dis_val: "));
      Serial.println(dis_val_mm);
#endif

      // ======================================================
      // 2. メインの動作
      // ======================================================
      // mode による場合分け
      switch (mode_val) {
        // 動作モード, nakano
        case 1:
          {
#ifdef DEBUG
            Serial.print("mode: ");
            Serial.println("move mode");
#endif
            // 1kHzの時間ゲート（常にここで回す）
            unsigned long now = micros();
            if ((long)(now - t_next) < 0)
              return;                                    // まだ周期に満たないなら何もしない
            float dt = (float)UPDATE_PERIOD_US * 1e-6f;  // 例: 0.001
            t_next += UPDATE_PERIOD_US;                  // 次の実行時刻へ

            // 移動計算
            float x_cmd = 0.0f, y_cmd = 0.0f;
            computeCommandRates(x_cmd, y_cmd);  // 角度優先→距離

            // 移動実行
            driveVelocity(x_cmd, y_cmd, dt);
            break;
          }

        // 打つモード, takemichi
        case 2:
          {
#ifdef DEBUG
            Serial.print("mode: ");
            Serial.println("hitting mode");
#endif
            int val;
            val = (257 - dis_val_mm) / 2;
            angleA = val;
            // Serial.print(F("ミニサーボを "));
            // Serial.print(angleA);
            // Serial.println(F(" 度に移動"));
            delay(1000);
            servoB.write(150);  // ミニサーボをロック位置にセット
            delay(1000);
            servoA.write(angleA);
            delay(2000);
            servoB.write(40);
            delay(2000);
            servoA.write(180);
            break;
          }

        default:
#ifdef DEBUG
          Serial.print("mode: ");
          Serial.println("Non mode");
#endif
      }
    }
  }
}  // void loop()