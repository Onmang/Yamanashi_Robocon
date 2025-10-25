/*
 * アナログ値の読み取りと、しきい値でのLED制御
 * * 元のコードから以下を変更:
 * 1. A0の値をシリアルモニタに表示する
 * 2. A0の値がしきい値(threshold)を超えたらLEDを点灯、
 * 超えなければ消灯する
 */

int sensorPin = A0;    // ポテンショメータ(可変抵抗)などを接続するピン
int ledPin = 13;       // LEDを接続するピン
int sensorValue = 0;   // センサーの値を格納する変数

//  LEDを点灯させる しきい値 
// sensorValue がこの値(0〜1023)より大きくなったらLEDが光ります。
const int threshold = 500; 

void setup() {
  // LEDピンを出力に設定
  pinMode(ledPin, OUTPUT);
  
  //  シリアルモニタを開始 (PCで値を見るため) 
  Serial.begin(9600); 
}

void loop() {
  // 1. センサーの値を読み取る (0〜1023の値が入る)
  sensorValue = analogRead(sensorPin);
  
  // 2.  読み取った値をシリアルモニタに表示する 
  Serial.println(sensorValue);

  // 3.  しきい値と比較してLEDを制御する 
  if (sensorValue > threshold) {
    // しきい値より大きければ、LEDを点灯
    digitalWrite(ledPin, HIGH);
  } else {
    // しきい値以下なら、LEDを消灯
    digitalWrite(ledPin, LOW);
  }
  
  // 値の表示が速すぎないように、少し待つ
  delay(100); 
}