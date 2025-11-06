#include <Arduino.h>
#include <math.h>
#include <Servo.h>
#include "nakano.h"

//#define DEBUG

//============== SERVO: takemichi ==================//
Servo servoA;
Servo servoB;
Servo servoC;
int angleA = 180;
int angleB = 40;
int angleC = 105;

//============== MOVE: Nakano ==================//
// 長いためヘッダーファイルに分割


//超音波
int sensorPin = A1;
int ledPin1 = 13;
int ledPin2 = 12;
int ledPin3 = 11;
int i = 0;
int ave = 0;
unsigned long now = 0;

const int threshold_MAX = 500;
const int threshold_ave = 450;

const int MAX_SAMPLES = 100;      // 500msで入れたい最大サンプル数
int buf[MAX_SAMPLES];
int sampleCount = 0;

int maxValue = 0;
unsigned long lastMs = 0;


//  LEDを点灯させる しきい値 
// sensorValue がこの値(0〜1023)より大きくなったらLEDが光ります。
const int threshold = 140; 

// その他
const int EXPECTED_LEN = 10;

void setup() {
  // シリアル開始（※コメントと実値を一致させています）
  Serial.begin(115200);
  Serial.setTimeout(50);  // readStringUntilの待ち時間（ms）

  // takemichi servo setup
  servoA.attach(11);   // デカサーボ
  servoB.attach(10);  // ミニサーボ
  servoC.attach(9);  // 角度サーボ

  servoA.write(angleA);
  delay(1500);
  servoB.write(angleB);
  delay(1000);
  servoB.write(150); //ミニサーボをロック位置にセット
  delay(1000);
  servoA.write(30);
  delay(2000);

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

  pinMode(ledPin1, OUTPUT);
  pinMode(ledPin2, OUTPUT);
  pinMode(ledPin3, OUTPUT);
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
      float x_cmd = 0.0f, y_cmd = 0.0f;
      computeCommandRates(x_cmd, y_cmd); // 角度優先→距離
      driveVelocity(x_cmd, y_cmd, dt);   // ステップ吐く
      break;
    }

    case 2: {
      int val = (257 - dis_val_mm) / 2;
      angleA = constrain(val, 130, 0);
  
      // ひとまずそのままやるならこう
      delay(1000);
      servoA.write(angleA);
      delay(2000);
      servoB.write(40);
      delay(4000);  
      servoA.write(180);
      delay(4000);
      servoB.write(150);
      delay(1000);
      servoA.write(30);
      break;
    }

    case 3: {
      int v = analogRead(sensorPin);

      // maxの更新
      if (v > maxValue) {
        maxValue = v;
      }

      if (v > 420 && sampleCount < MAX_SAMPLES) {
        buf[sampleCount] = v;
        sampleCount++;
      }

      now = millis();
      if (now - lastMs >= 200) {
        //int ave = 0;
        if (sampleCount > 0) {
          // 雑に平均を出す
          long sum = 0;
          for (int k = 0; k < sampleCount; k++) {
            sum += buf[k];
          }
          int roughAve = sum / sampleCount;

          // 平均から離れてるのを除外し再平均
          sum = 0;
          int validCount = 0;
          const int ALLOW_DIFF = 50;   // これより離れてたら外れとみなす（要調整）
          for (int k = 0; k < sampleCount; k++) {
            if (abs(buf[k] - roughAve) <= ALLOW_DIFF) {
              sum += buf[k];
              validCount++;
            }
          }

          if (validCount > 0) {
            ave = sum / validCount;
          } else {
            ave = 0;   // 全部外れた場合
          }
        }

      
      // 出力 
      Serial.print("MAX:"); Serial.print(maxValue);
      if(maxValue>threshold_MAX){
      Serial.print("*");
      }
      Serial.print(" AVE:"); Serial.print(ave);
      if(ave>threshold_ave){
      Serial.print("*");
      }
      Serial.print(" N:");   Serial.print(sampleCount);  

      //認識
      if (maxValue > threshold_MAX && ave > threshold_ave) {
        //Serial.print("  !");
        //digitalWrite(ledPin1, HIGH);
        i = 1;
      }else{
        //digitalWrite(ledPin1, LOW);
        //digitalWrite(ledPin2, LOW);
        //x_cmd = ROT_CCW_X_SIGN * 0.10f;
        //y_cmd = ROT_CCW_Y_SIGN * (-0.10f); 
        //driveVelocity(x_cmd, y_cmd, dt);
        i=0;
      }


      Serial.println();

      // 片付け
      maxValue = 0;
      sampleCount = 0;
      lastMs = now;
      }

      if(i==0){
        x_cmd = ROT_CCW_X_SIGN * 0.10f;
        y_cmd = ROT_CCW_Y_SIGN * (-0.10f); 
        driveVelocity(x_cmd, y_cmd, dt);
        delay(dt);
      }
      
      //delay(5); 
      break;         
    } 




    case 4: {
     moveAbsolute((float)angle_deg, (float)dis_val_mm);
     mode_val = 0; // 1回だけ動作して停止
     break;
    }
 

    default: {
      // mode_valが1でも2でもない → 何もしない（停止）
      break;
    }
  }
}
