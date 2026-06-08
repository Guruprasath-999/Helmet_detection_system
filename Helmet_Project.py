import cv2
from ultralytics import YOLO
import serial
import time

# ---------------- SERIAL SETTINGS ----------------
arduino_port = "COM8"   # Change this
baud_rate = 9600

try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)
    print("Arduino Connected Successfully!")
except:
    print("ERROR: Arduino not connected or wrong COM port!")
    arduino = None

# ---------------- LOAD YOLO MODEL ----------------
model = YOLO("helmet.pt")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # use 0 or 1

helmet_last = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working!")
        break

    helmet_found = False

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]

            if name.lower() == "helmet" and conf > 0.2:
                helmet_found = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Send data only when change happens
    if arduino is not None and helmet_found != helmet_last:
        if helmet_found:
            arduino.write(b'1')
            print("Helmet detected -> Relay allowed")
        else:
            arduino.write(b'0')
            print("No helmet -> Relay blocked + buzzer ON")

        helmet_last = helmet_found

    # Display status
    if helmet_found:
        cv2.putText(frame, "HELMET DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "NO HELMET", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Helmet Detection Project", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if arduino is not None:
    arduino.close()