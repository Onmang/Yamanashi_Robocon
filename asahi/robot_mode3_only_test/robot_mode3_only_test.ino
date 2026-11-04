
const int PIN26 = 26;
const int PIN27 = 27;

void setup() {
  Serial.begin(115200);

  pinMode(PIN26, OUTPUT);
  pinMode(PIN27, OUTPUT);

  digitalWrite(PIN26, LOW);
  digitalWrite(PIN27, LOW);
}

void loop() {
  // D26だけHIGH
  //Serial.println("D26 HIGH / D27 LOW");
  //digitalWrite(PIN26, HIGH);
  //digitalWrite(PIN27, LOW);
  //delay(3000);

  // D27だけHIGH
  Serial.println("D26 LOW / D27 HIGH");
  digitalWrite(PIN26, LOW);
  digitalWrite(PIN27, HIGH);
  delay(3000);

  // 両方LOW
  Serial.println("D26 LOW / D27 LOW");
  digitalWrite(PIN26, LOW);
  digitalWrite(PIN27, LOW);
  delay(3000);
}
