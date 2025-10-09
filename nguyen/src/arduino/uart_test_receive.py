import serial
import time

# ls -l /dev/ | grep tty
# Arduinoが接続されているシリアルポートとボーレートを設定
serial_port = '/dev/ttyACM0'
baud_rate = 9600

try:
    # シリアルポートを開く
    ser = serial.Serial(serial_port, baud_rate)
    print("Serial Port was opened: " + serial_port)

    while True:
        # データが利用可能であれば読み取る
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print("Received Data: " + line)

except KeyboardInterrupt:
    # Ctrl+Cが押されたら終了
    print("Program is terminating")
    ser.close()

except Exception as e:
    # その他のエラー
    print("Error occurred: " + str(e))
    ser.close()