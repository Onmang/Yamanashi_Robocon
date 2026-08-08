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
//int ledPin1 = 13;
//int ledPin2 = 12;
//int ledPin3 = 11;
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

//  pinMode(ledPin1, OUTPUT);
//  pinMode(ledPin2, OUTPUT);
//  pinMode(ledPin3, OUTPUT);
}

// ==== 共通ユーティリティ ====

// シリアルバッファを空にする（古い命令を捨てる）
void flushSerial()
{
  while (Serial.available()) {
    Serial.read();
  }
}

// ステッピングの状態リセット＆停止
void resetStepper()
{
  x_step_accum = 0.0f;
  y_step_accum = 0.0f;
  driveVelocity(0.0f, 0.0f, (float)UPDATE_PERIOD_US * 1e-6f);
  t_next = micros() + UPDATE_PERIOD_US;  // 次周期を「今基準」に
}

// 常に「最後の1本だけ」読む版
void readLatestCommand()
{
  if (!Serial.available()) return;

  String last = "";

  // バッファに溜まっている行を全部読む
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == EXPECTED_LEN) {
      bool all_digit = true;
      for (uint8_t i = 0; i < line.length(); i++) {
        if (!isDigit(line[i])) {
          all_digit = false;
          break;
        }
      }
      if (all_digit) {
        last = line;  // 有効なものだけ上書き → 最後の1本だけ残る
      }
    }
  }

  if (last == "") return;

  int mode = last.substring(0, 1).toInt();

  int sign_flag_deg = last.substring(1, 2).toInt();
  int angle_abs     = last.substring(2, 5).toInt();
  int angle_signed  = (sign_flag_deg == 1) ? angle_abs : -angle_abs;

  int sign_flag_dis = last.substring(5, 6).toInt();
  int dis_abs       = last.substring(6, 10).toInt();
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


void loop() {

  // ---------------------------------
  // 1. シリアル受信
  // ---------------------------------
  //  if (Serial.available()) {
  //    String receivedData = Serial.readStringUntil('\n');
  //    receivedData.trim();
  //#ifdef DEBUG
  //    Serial.print("Received Data: ");
  //    Serial.println(receivedData);
  //#endif
  //

  readLatestCommand();
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
      digitalWrite(EN_PIN, HIGH);  // 無効化（モータOFF）
      int val = (2570 - dis_val_mm) / 20;
      angleA = constrain(val, 30, 130);
  
      // ひとまずそのままやるならこう
      servoA.write(angleA);//セット
      delay(1000);
      servoB.write(40);//打つ
      delay(1000);  
      servoA.write(30);//振り上げる
      delay(2000);
      servoB.write(150);//セット
      delay(2000);
      servoA.write(150);//落ち着く
      delay(1000);
      servoB.write(40);//解放
      delay(1000);
      servoA.write(180);//じゅんび
      delay(3000);
      servoB.write(150);//セット
      delay(1000);
      servoA.write(30);//振り上げ
      delay(1000);
      // ★ ブロッキング後の暴走防止 ★
      flushSerial();    // mode2中に溜まったゴミコマンドを捨てる
      resetStepper();   // ステッピング内部状態リセット
      mode_val = 0;     // 次のコマンドが来るまで待機
      digitalWrite(EN_PIN, LOW);   // 再度有効化
      delay(1000);
      Serial.println("DONE");
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
      // 絶対値移動が終わったら状態クリア
      flushSerial();
      resetStepper();
      mode_val = 0;
     break;
    }
 

    default: {
      // mode_valが1でも2でもない → 何もしない（停止）
      break;
    }
  }
}
