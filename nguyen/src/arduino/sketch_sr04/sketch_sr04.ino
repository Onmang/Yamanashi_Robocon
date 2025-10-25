/*
 * HC-SR04 送信専用Arduino
 * センサーAを接続
 */

#define trigPin 9  // センサーAのTrigピン
#define LED_PIN 13

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  // 100msごとに超音波を発射

  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // 送信の合図 (10us)
  digitalWrite(trigPin, HIGH);
  digitalWrite(LED_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  digitalWrite(LED_PIN, LOW);

  // 100ms待つ (1秒間に10回送信)
  delay(100);
}