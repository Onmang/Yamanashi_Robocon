void setup() {
  // シリアル通信を9600bpsで開始
  Serial.begin(9600);
}

void loop() {
  // "Hello, Raspberry Pi"を送信
  // Serial.println("Hello, Raspberry Pi");
  // 2秒待機
  // delay(2000);

  // Raspberry Piからのデータを受信して表示
  if (Serial.available() > 0) {                           
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
      } else {
        Serial.println("Error: Received data contains non-digit characters");
      }
    } else {
      Serial.print("Error: Unexpected data length (expected ");
      Serial.print(EXPECTED_LEN);
      Serial.print(") got ");
      Serial.println(receivedData.length());
    }
  }
}
