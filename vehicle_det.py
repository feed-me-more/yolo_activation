import cv2
from ultralytics import YOLO
import time
import numpy as np
from pathlib import Path
import pickle
from compare_accuracy import average, compare_detections


model = YOLO("yolov8n.pt")
source = "road_traffic.mp4"  
frame_id = 0
start_time = 0
end_time = 0    
max_inf_time = 0
max_inf_frame = -1
prev_inf_time = 0
result_metric = []

cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {source}")

VEHICLE_CLASSES = {1, 2, 3, 5, 7}

with open("reference_det_set_640_360.pkl", "rb") as f:
    reference_dets = pickle.load(f)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(
    "processed_video.mp4",
    fourcc,
    30,
    (640, 360)
)

start_time = time.perf_counter()

while True:
    ret, frame = cap.read()                             
    frame_start_time = time.perf_counter()

    if not ret:
        break

    results = model(frame, verbose=False)
    recv_detections = []

    for box in results[0].boxes:
        cls = int(box.cls.item())
        
        if cls in VEHICLE_CLASSES:

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf.item())

            recv_detections.append([x1, y1, x2, y2, conf, cls])

            label = f"{model.names[cls]} {conf:.2f}"

            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1e3

            cv2.rectangle(
                frame, 
                (x1, y1), 
                (x2, y2), 
                # 0.7 * (0, 255, 0) + 0.3 * (255, 0, 0),
                (0, 255, 0),
                1,
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

            # frame_end_time = time.perf_counter()
            # inference_time_ms = (frame_end_time - frame_start_time) * 1e3

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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    reference_det = reference_dets.get(frame_id,np.zeros((0, 6), dtype=np.float32))
    recv_detection = np.array(recv_detections, dtype=np.float32) if recv_detections else np.zeros((0, 6), dtype=np.float32)

    if (inference_time_ms > max_inf_time) and frame_id > 0:
        max_inf_time = inference_time_ms
        max_inf_frame = frame_id

    

    # annotated = cv2.resize(annotated, (640, 360))
    video.write(annotated)

    metrics = compare_detections(recv_detection, reference_det)

    result_metric.append({"frame_id": frame_id + 1, **metrics})

    frame_id += 1

end_time = time.perf_counter()
total_time = end_time - start_time
    
print(f"Processed {frame_id} frames in {total_time:.3f} s ") 
print(f"Average inference time: {total_time / (frame_id) * 1e3:.3f} ms per frame")
print(f"Max inference time: {max_inf_time:.1f} ms on frame {max_inf_frame}")

average_metrics = average(result_metric)
print(f"Average metrics over {len(result_metric)} frames: {average_metrics}")

cap.release()
cv2.destroyAllWindows()