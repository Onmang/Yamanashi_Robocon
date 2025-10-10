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

    # Example sending loop: build numeric message in format M A A A D D D D\n
    # where M = mode (1 digit), AAA = angle (3 digits, zero-padded), DDDD = distance (4 digits, zero-padded)
    while True:
        # Example values; replace with your actual logic for mode/angle/dis
        mode = 1
        angle = value % 1000       # 0..999
        dis = (value // 1000) % 10000  # 0..9999

        # Format with zero-padding to fixed widths so Arduino can parse consistently
        msg = f"{mode}{angle:03d}{dis:04d}\n"

        try:
            ser.write(msg.encode('ascii'))
            print(f"Sent: {msg.strip()}")
        except Exception as e:
            print("Failed to write to serial: " + str(e))

        value += 1
        # avoid busy loop
        # 100 msec
        time.sleep(0.1)

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
