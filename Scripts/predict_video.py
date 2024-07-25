import cv2
import os

from ultralytics import YOLO

VIDEOS_DIR = './videos'
OUTPUTS_DIR = './outputs'
VIDEO_NAME = 'test_0.mp4'

video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
video_path_out = video_path.replace('.mp4', '_out.mp4')

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

ret, frame = cap.read()
if ret:
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join(OUTPUTS_DIR, 'best.pt')
model = YOLO(model_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Adjust the call according to the correct API usage
    results = model(frame)  # Ensure that the API call here is correct based on YOLO documentation

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(frame, f'{cls} {conf:.2f}', (int(xyxy[0]), int(xyxy[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
