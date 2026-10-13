from ultralytics import YOLO

model = YOLO('C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning-2/weights/best.pt')
imgs = [
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/test/6.jpg',
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/val/1.jpg',
]
r = model.predict(imgs, device=0, conf=0.05, save=True,
                  project='C:/Users/28687/Desktop/X/yolo_train/runs/predict',
                  name='gpu_test_conf05', exist_ok=True, verbose=False)
for x in r:
    dets = [(model.names[int(b.cls)], round(float(b.conf), 3)) for b in x.boxes]
    name = x.path.replace('\\', '/').split('/')[-1]
    print(name, '->', dets if dets else 'conf>=0.05 仍无目标')
print('结果图保存在:', r[0].save_dir)
