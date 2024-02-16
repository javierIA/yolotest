from ultralytics import YOLO

uri="rtsp://admin:admin@10.210.50.35:8554/profile0"
#rtspt://admin:H2FuiDp4@10.33.1.11:8554/profile0
#uri="TESTVIDEO.mp4"
model = YOLO('best.pt')  # pretrained YOLOv8n model
# Run batched inference on a list of images
results = model(uri,show=True,save=True) 