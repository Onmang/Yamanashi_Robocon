import serial
import time

# ls -l /dev/ | grep tty
ARDUINO = True
if ARDUINO:
    global ser

    serial_port = "/dev/ttyUSB0"  # arduino UNO
    # serial_port = '/dev/ttyACM0'
    # takemichi arduino nano evry
    baud_rate = 115200  # 9600, 115200
    ser = serial.Serial(
        serial_port,
        baud_rate,  # できれば 115200 を推奨
        timeout=1,  # 読み取りは非ブロッキング（読みはしてないが安全）
        write_timeout=0,  # 書き込みもブロッキングしない
    )
    time.sleep(2.0)  # リセット待ち 単位：sec
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("Serial Port was opened:", serial_port)

try:
    mode = 1
    angle_val = 10  # degrees
    distance_val = 200  # mm
    send_time = 1/15 # 15 fps
    counter = 1

    while True:
        
        if counter >= 15:
            print(f"Sent {counter} messages, exiting.")
            break
        
        # データが利用可能であれば読み取る
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print("Received Data: " + line)
            
            value = int(line)  # 受け取ったデータを整数に変換してvalueにセット
            if value % 2 == 0:
                print(value)
                if value > 100:
                    print("Value exceeded 100, exiting.")
                    break   
        
        
        # Example values; replace with your actual logic for mode/angle/dis
        angle = angle_val % 1000       # 0..999
        dis = (distance_val // 1000) % 100000  # 0..99999

        # Format with zero-padding to fixed widths so Arduino can parse consistently
        msg = f"{mode}{angle:04d}{dis:05d}\n"

        try:
            ser.write(msg.encode('ascii'))
            counter += 1
            print(f"Sent {counter}: {msg.strip()}")
        except Exception as e:
            print("Failed to write to serial: " + str(e))

        # avoid busy loop
        time.sleep(send_time)

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
