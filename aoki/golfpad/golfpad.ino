#include <Servo.h>

Servo servoA;
Servo servoB;
Servo servoC;

int angleA = 180;
int angleB = 40;
int angleC = 105;

int targetIsA = 1;  // trueなら次はAを動かす、falseなら次はB

void setup() {
  Serial.begin(9600);

  servoA.attach(9);    // デカサーボ
  servoB.attach(10);   // ミニサーボ
  servoC.attach(11);   //角度サーボ

  servoA.write(angleA);
  servoB.write(angleB);
  servoC.write(angleC);

  Serial.println(F("サーボ制御開始"));
  Serial.println(F("最初の入力はデカサーボ、次の入力はミニサーボ、次は角度サーボ、交互に入力します"));
  Serial.println(F("0〜180の角度を入力してください"));
}

void loop() {
  if (Serial.available()) {
    String receivedData = Serial.readStringUntil('\n');
    receivedData.trim(); // remove CR/LF and spaces
    Serial.print("Received Data: ");
    Serial.println(receivedData);

    // Expecting a numeric string like "12223333" (1 + 3 + 4 = 8 digits?)
    // According to the user: mode:1 digit, angle:3 digits, dis:4 digits
    // Total length expected = 1 + 3 + 4 = 8
    const int EXPECTED_LEN = 8;

    if (receivedData.length() == EXPECTED_LEN) {
      bool allDigits = true;
      for (unsigned int i = 0; i < receivedData.length(); ++i) {
        if (!isDigit(receivedData.charAt(i))) {
          allDigits = false;
          break;
        }
      }

      if (allDigits) {
        int mode_val = receivedData.substring(0, 1).toInt();
        int angle_val = receivedData.substring(1, 4).toInt();
        int dis_val = receivedData.substring(4, 8).toInt();

        Serial.print("mode_val: ");
        Serial.println(mode_val);
        Serial.print("angle_val: ");
        Serial.println(angle_val);
        Serial.print("dis_val: ");
        Serial.println(dis_val);

        int targetIsA = mode_val;

        //int val=map(dis_val,0,500,20,180);
        //val=180-val;
        int val;
        val=(257-dis_val)/2;

        if (targetIsA==2) {
        Serial.write('2'); //シリアル通信：受信 
        angleA = val;
        //Serial.print(F("ミニサーボを "));
        //Serial.print(angleA);
        //Serial.println(F(" 度に移動"));
        delay(1000);
        servoB.write(150); //ミニサーボをロック位置にセット
        delay(1000);
        servoA.write(angleA);
        delay(2000);
        servoB.write(40);
        delay(2000);
        servoA.write(180);
        targetIsA = 0;  // 角度をセット
        Serial.write('0'); //シリアル通信：完了 
        } else{
        targetIsA = 0;
        }

      } else {
        Serial.println("Error: Received data contains non-digit characters");
      }
    } else {
      Serial.print("Error: Unexpected data length (expected ");
      Serial.print(EXPECTED_LEN);
      Serial.print(") got ");
      Serial.println(receivedData.length());
    }

    // 指令がないときは何もしない → 角度保持
  }
}
