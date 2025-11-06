# 割込み、押されたらcounterをリセット
import RPi.GPIO as GPIO
import time
import sys

Sw_pin = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(Sw_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

counter = 0  # グローバル変数として定義

# 割込み時に呼ばれるコールバック関数
def sw_callback(channel):
    global counter  # 外のcounterを使う宣言
    counter = 0
    print("スイッチが押された！ counterリセット！")

# 割込み設定（立ち上がり検出＋チャタリング防止）
GPIO.add_event_detect(Sw_pin, GPIO.RISING, callback=sw_callback, bouncetime=500)

try:
    while True:
        print(f"[Debug] counter={counter}")
        time.sleep(1)
        counter += 1

except KeyboardInterrupt:
    GPIO.cleanup()
    sys.exit()
