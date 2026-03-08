#define SERVO_PIN 9

void setup() {
  pinMode(SERVO_PIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(1450);
  digitalWrite(SERVO_PIN, LOW);
  delay(1000);
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(2400);
  digitalWrite(SERVO_PIN, LOW);
  delay(1000);

}
