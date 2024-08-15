from ultralytics import YOLO


model = YOLO("yolov8n.pt")

metrics = model.val(data="data.yaml")

print(metrics.box.map)