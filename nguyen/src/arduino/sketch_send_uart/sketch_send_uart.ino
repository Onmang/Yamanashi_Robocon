void setup() {
  // シリアル通信を9600bpsで開始
  Serial.begin(115200);
  Serial.setTimeout(50); // 改行までの待ち時間（ms）
}

void loop() {

  // Raspberry Piからのデータを受信して表示
  if (Serial.available() > 0) {                           
    String receivedData = Serial.readStringUntil('\n');
    receivedData.trim(); // remove CR/LF and spaces
    Serial.print("Received Data: ");
    Serial.println(receivedData);

    // Total length expected 
    const int EXPECTED_LEN = 10;

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
        // 角度の符号フラグと絶対値
        int sign_flag_deg = receivedData.substring(1, 2).toInt(); // 0 or 1
        int angle_abs = receivedData.substring(2, 5).toInt(); // 000..999
        if (sign_flag_deg != 0 && sign_flag_deg != 1) {
          Serial.println("Error: invalid sign flag (must be 0 or 1)");
          return;
        }
        int angle_deg = (sign_flag_deg == 1) ? angle_abs : -angle_abs;

        // 距離の符号フラグと絶対値
        int sign_flag_dis = receivedData.substring(5, 6).toInt(); // 0 or 1 (not used here)
        if (sign_flag_dis != 0 && sign_flag_dis != 1) {
          Serial.println("Error: invalid sign flag (must be 0 or 1)");
          return;
        }
        int dis_abs = receivedData.substring(6, 10).toInt();
        int dis_val = (sign_flag_dis == 1) ? dis_abs : -dis_abs;

        Serial.print("mode_val: ");
        Serial.println(mode_val);
        Serial.print("angle_val: ");
        Serial.println(angle_deg);
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
