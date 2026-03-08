#include <Servo.h>

#define SERVO_PIN 9

Servo myservo;  // サーボオブジェクトを作成

void setup() {
  // 適切な角度はここを調整する pulse_min と pulse_max 
  // 2025/11/11 調整 SG90
  int pulse_min = 490; // default 544us
  int pulse_max = 2490; // default 2400us
  myservo.attach(SERVO_PIN, pulse_min, pulse_max);  // サーボを9番ピンに接続


  // シリアル通信を開始 (ボーレートは 9600 bps)
  Serial.begin(9600);
  myservo.write(0);
  
  // 準備ができたらメッセージを表示
  Serial.println("シリアルモニタから角度 (0-180) を入力してください。");
  Serial.println("例: 90");
}

void loop() {
  // シリアルポートにデータが送信されているか確認
  if (Serial.available() > 0) {
    
    // シリアルから送られてきた文字列を整数 (integer) に変換して読み取る
    // "90" と入力して送信すると、 90 という数値として読み込まれます
    int angle = Serial.parseInt();

    // サーボに角度を指示
    // Servoライブラリは、0未満の値は0、180より大きい値は180として自動的に処理します
    myservo.write(angle);
    delay(1000);

    // どの角度に設定したかをシリアルモニタにフィードバック表示
    Serial.print("サーボを ");
    Serial.print(angle);
    Serial.println(" 度に設定しました。");

    // シリアルバッファに残っている可能性のある改行コードなどを読み飛ばす
    // これがないと、改行コードを「0」として読み取ってしまう場合があります
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}