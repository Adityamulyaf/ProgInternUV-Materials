from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolov5n.pt")

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera ngga bisa dibuka woi, kenapa yhhh?")
    exit()

prev_time = time.time()

while True:

    # Ambil frame dari kamera
    ret, frame = cap.read()

    if not ret:
        print("Frame ngga kebaca")
        break

    # Object detection
    #results = model(frame, verbose=False)
    results = model(frame, conf=0.4, verbose=False)

    print(results[0].boxes)

    # Gambar bounding box
    frame = results[0].plot()

    # colors = {
    # 0: (0,255,0),
    # 1: (0,0,255)
    # }

    # for box in results[0].boxes:

    #     x1, y1, x2, y2 = box.xyxy[0]
    #     conf = float(box.conf[0])
    #     cls = int(box.cls[0])

    #     label = model.names[cls]
    #     color = colors.get(cls, (255,255,255))

    #     cv2.rectangle(frame,
    #                 (int(x1), int(y1)),
    #                 (int(x2), int(y2)),
    #                 color, 2)

    #     cv2.putText(frame,
    #                 f"{label} {conf:.2f}",
    #                 (int(x1), int(y1)-10),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.6,
    #                 color,
    #                 2)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Tampilkan FPS di layar
    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Tampilka hasil
    cv2.imshow("Detection YOLOv5", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()