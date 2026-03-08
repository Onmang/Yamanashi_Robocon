#include <Servo.h>
Servo motor_servo;

void setup() {
  motor_servo.attach(9,500,2400);
}

void loop() {
  motor_servo.writeMicroseconds(500);
  delay(1000);

  motor_servo.writeMicroseconds(1500); // 90 deg
  delay(1000);
   
  motor_servo.writeMicroseconds(2400);
  delay(1000);
}
