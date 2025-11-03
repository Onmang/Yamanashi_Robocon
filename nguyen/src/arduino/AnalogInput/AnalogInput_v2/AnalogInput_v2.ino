/*
 * アナログ値の読み取りと、しきい値制御
 * * 元のコード:
 * 1. A0の値をシリアルモニタに表示する
 * 2. A0の値がしきい値(threshold)を超えたらLEDを点灯、
 * * 変更要素：
 * 1. サンプリング周期5msでデータを取得
 * 2. 500ms間サンプルを取得し、最大値と平均値を出力
 * 3. 最大及び平均から３段階で方向の判定を行う
 */
int sensorPin = A0;
int ledPin1 = 13;
int ledPin2 = 12;
int ledPin3 = 11;

const int threshold_MAX = 550;
const int threshold_ave = 540;

const int MAX_SAMPLES = 100;      // 500msで入れたい最大サンプル数
int buf[MAX_SAMPLES];
int sampleCount = 0;

int maxValue = 0;
unsigned long lastMs = 0;

void setup() {
  pinMode(ledPin1, OUTPUT);
  pinMode(ledPin2, OUTPUT);
  pinMode(ledPin3, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  int v = analogRead(sensorPin);

  // maxの更新
  if (v > maxValue) {
    maxValue = v;
  }

  if (v > 540 && sampleCount < MAX_SAMPLES) {
    buf[sampleCount] = v;
    sampleCount++;
  }

  unsigned long now = millis();
  if (now - lastMs >= 1000) {
    int ave = 0;
    if (sampleCount > 0) {
      // 雑に平均を出す
      long sum = 0;
      for (int k = 0; k < sampleCount; k++) {
        sum += buf[k];
      }
      int roughAve = sum / sampleCount;

      // 平均から離れてるのを除外
      sum = 0;
      int validCount = 0;
      const int ALLOW_DIFF = 50;   // これより離れてたら外れとみなす（要調整）
      for (int k = 0; k < sampleCount; k++) {
        if (abs(buf[k] - roughAve) <= ALLOW_DIFF) {
          sum += buf[k];
          validCount++;
        }
      }

      if (validCount > 0) {
        ave = sum / validCount;
      } else {
        ave = 0;   // 全部外れた場合
      }
    }

    // 出力
    Serial.print("MAX:"); Serial.print(maxValue);
    if(maxValue>threshold_MAX){
      Serial.print("*");
    }
    Serial.print(" AVE:"); Serial.print(ave);
    if(ave>threshold_ave){
      Serial.print("*");
    }
    Serial.print(" N:");   Serial.print(sampleCount);  

    // 判定
    if (maxValue > threshold_MAX + 20 && ave > threshold_ave + 40) {
      Serial.print("  !!!");
      digitalWrite(ledPin1, HIGH);
      digitalWrite(ledPin2, HIGH);
      digitalWrite(ledPin3, HIGH);
    } else if (maxValue > threshold_MAX + 10 && ave > threshold_ave + 20) {
      Serial.print("  !!");
      digitalWrite(ledPin1, HIGH);
      digitalWrite(ledPin2, HIGH);
      digitalWrite(ledPin3, LOW);
    } else if (maxValue > threshold_MAX && ave > threshold_ave) {
      Serial.print("  !");
      digitalWrite(ledPin1, HIGH);
      digitalWrite(ledPin2, LOW);
      digitalWrite(ledPin3, LOW);
    }else{
      digitalWrite(ledPin1, LOW);
      digitalWrite(ledPin2, LOW);
      digitalWrite(ledPin3, LOW);
    }

    Serial.println();

    // 片付け
    maxValue = 0;
    sampleCount = 0;
    lastMs = now;
  }

  delay(5);
}
