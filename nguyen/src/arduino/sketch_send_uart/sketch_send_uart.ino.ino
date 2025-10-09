void setup() {
  // シリアル通信を9600bpsで開始
  Serial.begin(9600);
}

void loop() {
  // "Hello, Raspberry Pi"を送信
  Serial.println("Hello, Raspberry Pi");
  // 2秒待機
  delay(2000);
}