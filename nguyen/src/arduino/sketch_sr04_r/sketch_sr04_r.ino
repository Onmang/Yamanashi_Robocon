/*
 * HC-SR04 受信専用Arduino
 * センサーBを接続
 */
#define trigPin_B 9
#define echoPin_B 10
#define LED_PIN 13

void setup() {
  Serial.begin(9600);
  pinMode(trigPin_B, OUTPUT);
  pinMode(echoPin_B, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void loop() {

  digitalWrite(trigPin_B, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin_B, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin_B, LOW);

  // 2. 直後に受信待機を開始 (タイムアウトは25msに設定)
  // (もしセンサーAからの音が届けば、ここでdurationに値が入るはず)
  // (もし何も受信しなければ 0 になる)
  long duration = pulseIn(echoPin_B, HIGH, 110000);  // タイムアウトを110msに延長

  // 3. 判定
  if (duration > 0) {
    Serial.println("Recieved");
    digitalWrite(LED_PIN, HIGH);
  } else {
    Serial.println("Non-recieved");
    digitalWrite(LED_PIN, LOW);
  }

  // loop()がすぐに回りすぎないように少しだけ待つ
  delay(10);
}