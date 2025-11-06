#include <Servo.h>

Servo servoA;
Servo servoB;
Servo servoC;

int angleA = 180;
int angleB = 40;
int angleC = 105;

volatile bool risingFlag = false;
volatile bool fallingFlag = false;
int prevState = LOW;  // 直前の状態（初期値LOW）

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  servoA.attach(9);    // デカサーボ
  servoB.attach(10);   // ミニサーボ
  //servoC.attach(9);   //角度サーボ

  servoA.write(angleA);
  delay(1500);
  servoB.write(angleB);

  delay(1000);
  servoB.write(150); //ミニサーボをロック位置にセット
  delay(1000);
  servoA.write(30);
  delay(2000);
}

void loop() {

  int state = digitalRead(13);
  
  if (prevState == LOW && state == HIGH) {
    risingFlag = true;
  } else if (prevState == HIGH && state == LOW){
    fallingFlag = true;
  }

  prevState = state; // 状態を記憶

  if (risingFlag) {
    //int val = (257 - dis_val_mm) / 2;
    //angleA = constrain(val, 130, 0);
    
    angleA=30;

    delay(1000);
    servoA.write(angleA);
    delay(2000);
    servoB.write(40);
    delay(1000);
    risingFlag = false;
    Serial.println("RISING edge detected");
  }
  if (fallingFlag) {
    delay(4000);  
    servoA.write(180);
    delay(4000);
    servoB.write(150);
    delay(1000);
    servoA.write(30);
    fallingFlag = false;
    Serial.println("FALLING edge detected");
  }
}
