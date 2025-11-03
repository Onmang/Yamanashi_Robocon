int counter = 0;
const int LED_PIN = 13;
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, HIGH);
  digitalWrite(LED_PIN, LOW);
  delay(10);
}
void loop() {
  counter++;
  if (counter % 2==0)
  {
    Serial.println(counter);
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
  }
}
