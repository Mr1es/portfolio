from ultralytics import YOLO
YOLO(r'C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning_gpu-2/weights/best.pt') \
    .predict(source=0, show=True, imgsz=640, conf=0.1)   # source=0 是摄像头，没摄像头就填视频路径