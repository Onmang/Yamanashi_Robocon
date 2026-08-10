import serial
import time

# arduino シリアル通信設定
ARDUINO = True  # True: シリアル通信ON, False: シリアル通信OFF
if ARDUINO:
    global ser
    #serial_port = "/dev/ttyUSB0"  # arduino UNO, ttyACM0
    serial_port = "COM5"  # Windows用。デバイスマネージャー > ポート(COMとLPT) で確認して変更

    baud_rate = 115200  # 9600, 115200
    ser = serial.Serial(
        serial_port,
        baud_rate,  # できれば 115200 を推奨
        timeout=0,  # 読み取りは非ブロッキング（読みはしてないが安全）
        write_timeout=0,  # 書き込みもブロッキングしない
    )
    time.sleep(2.0)  # リセット待ち 単位：sec
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("[Debug] Serial Port was opened:", serial_port)

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="ignore").rstrip()
            if not line:
                continue
            # print("Received:", line)
            if line == "DONE":
                print("Received:", line)
except KeyboardInterrupt:
    print("\n[FINISHED]")
finally:
    if ARDUINO and ser is not None:
        ser.close()

