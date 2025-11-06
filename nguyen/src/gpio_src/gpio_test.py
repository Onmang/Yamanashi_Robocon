# シンプルにgpio検出

#必要なモジュールをインポート
import RPi.GPIO as GPIO             #GPIO用のモジュールをインポート
import time                         #時間制御用のモジュールをインポート
import sys                          #sysモジュールをインポート

#ポート番号の定義
Sw_pin = 23                         #変数"Sw_pin"に23を格納

#GPIOの設定
GPIO.setmode(GPIO.BCM)              #GPIOのモードを"GPIO.BCM"に設定
#GPIO23を入力モードに設定してプルダウン抵抗を有効にする
GPIO.setup(Sw_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#while文で無限ループ
#GPIO23の入力を読み取る
while True:
    try:
        print(GPIO.input(Sw_pin))           #GPIO23が「ONで"1"」「OFFで"0"」
        time.sleep(1)                       #5秒間待つ
    except KeyboardInterrupt:               #Ctrl+Cキーが押された
        GPIO.cleanup()                      #GPIOをクリーンアップ
        sys.exit()                          #プログラムを終了