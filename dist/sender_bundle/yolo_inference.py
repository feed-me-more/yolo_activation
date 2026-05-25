from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("/home/yaswanth-ram-kumar/LT EDL/yolov8n.pt")
img_path = "bus.jpg"   # replace with your image path

res = model.predict(source=img_path, imgsz=640, device="cuda", verbose=False)[0]

annot = res.plot()[:, :, ::-1]  # BGR -> RGB for matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(annot)
plt.axis("off")
plt.title("YOLOv8n detections")
plt.show()

print("Detections:")
for row in res.boxes.data.cpu().numpy():
    x1, y1, x2, y2, conf, cls = row
    print(f"class={int(cls):2d}  conf={conf:.3f}  box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
