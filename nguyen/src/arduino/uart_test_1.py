import serial
import time

# ls -l /dev/ | grep tty
# Arduinoが接続されているシリアルポートとボーレートを設定
serial_port = '/dev/ttyACM0'
baud_rate = 9600
ser = None

try:
    # シリアルポートを開く
    ser = serial.Serial(serial_port, baud_rate)
    print("Serial Port was opened: " + serial_port)

    value = 0

    while True:
        # データが利用可能であれば読み取る
        # if ser.in_waiting > 0:
        #     line = ser.readline().decode('utf-8').rstrip()
        #     print("Received Data: " + line)

        # arduinoに送信
        ser.write(b'Hello from Raspberry Pi!\n')
        ser.write(str(value).encode())
        ser.write(b'\n')
        value += 1
        # avoid busy loop
        time.sleep(1)

except KeyboardInterrupt:
    # Ctrl+Cが押されたら終了
    print("Program is terminating")

except Exception as e:
    # その他のエラー
    print("Error occurred: " + str(e))

finally:
    # シリアルオブジェクトが作成されていれば安全にクローズ
    if ser is not None:
        try:
            if hasattr(ser, 'is_open'):
                if ser.is_open:
                    ser.close()
            else:
                ser.close()
            print("Serial port closed")
        except Exception as e:
            print("Failed to close serial port: " + str(e))