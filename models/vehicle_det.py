import cv2
from ultralytics import YOLO
import time


model = YOLO("yolov8n.pt")
source = "road_traffic.mp4"  
frame_id = 0
start_time = 0
end_time = 0    
max_inf_time = 0
max_inf_frame = -1
prev_inf_time = 0

cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {source}")

VEHICLE_CLASSES = {1, 2, 3, 5, 7}

start_time = time.perf_counter()

while True:
    ret, frame = cap.read()
    frame_start_time = time.perf_counter()

    if not ret:
        break

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls = int(box.cls.item())

        if cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf.item())

            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(
                frame, 
                (x1, y1), 
                (x2, y2), 
                # 0.7 * (0, 255, 0) + 0.3 * (255, 0, 0),
                (255, 0, 0),
                4,
            )
            # Class name and confidence
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1e3

            annotated = frame.copy()
            # Text on top-left corner with frame ID
            cv2.putText(
                annotated,
                f"frame {frame_id} in {inference_time_ms:.1f} ms",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    cv2.imshow("Vehicle Detection", annotated)

    if (inference_time_ms > max_inf_time) and frame_id > 0:
        max_inf_time = inference_time_ms
        max_inf_frame = frame_id
    frame_id += 1

    end_time = time.perf_counter()
    total_time = end_time - start_time

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"Processed {frame_id + 1} frames in {total_time * 1e3:.3f} ms ") 
print(f"Average inference time: {total_time / (frame_id + 1) * 1e3:.3f} ms per frame")
print(f"Max inference time: {max_inf_time:.1f} ms on frame {max_inf_frame}")

cap.release()
cv2.destroyAllWindows()