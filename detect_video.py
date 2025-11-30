from ultralytics import YOLO
import cv2


model = YOLO("runs/detect/train2/weights/best.pt")


input_video = "video/input.mp4"
output_video = "video/output.mp4"


cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25)

    for r in results:
        for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Готово! Відео з об’єктами збережено як", output_video)
