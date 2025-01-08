import cv2
import math
import winsound
from datetime import datetime
from ultralytics import YOLO

cap = cv2.VideoCapture("Videos/workers5.mp4")  # Change this to video file or camera stream

model = YOLO("best.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

def log_violation(violation_type):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open("violations.txt", "a") as file:
            file.write(f"{timestamp} - {violation_type} violation\n")
        print(f"Logged violation: {violation_type} at {timestamp}")  # Debugging message
    except Exception as e:
        print(f"Error logging violation: {e}")  # Catch and print any error

def trigger_alarm():
    winsound.Beep(1000, 1000)

alarm_triggered = False

while True:
    success, img = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    results = model(img, stream=True)
    missing_counts = {"Hardhat": 0, "Safety Vest": 0} 

    alarm_triggered = False  

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass not in ['Hardhat', 'Safety Vest', 'NO-Hardhat', 'NO-Safety Vest']:
                continue

            if conf > 0.5:
                if currentClass in ['NO-Hardhat', 'NO-Safety Vest']:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    if currentClass == 'NO-Hardhat':
                        missing_counts["Hardhat"] += 1
                        log_violation("NO-Hardhat")
                        if not alarm_triggered:
                            trigger_alarm()  
                            alarm_triggered = True
                    elif currentClass == 'NO-Safety Vest':
                        missing_counts["Safety Vest"] += 1
                        log_violation("NO-Safety Vest")
                        if not alarm_triggered:
                            trigger_alarm()  
                            alarm_triggered = True
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

   
    y_offset = 20 
    for item, count in missing_counts.items():
        if count > 0:
            cv2.putText(img, f"{count} Worker(s) not wearing {item}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
