from ultralytics import YOLO

W_BEST = 'C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning-2/weights/best.pt'
DATA = 'C:/Users/28687/Desktop/X/yolo_train/config/dataset.yaml'
imgs = [
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/test/6.jpg',
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/val/1.jpg',
]

model = YOLO(W_BEST)

print('=== 1) predict conf=0.001 全部输出 ===')
r = model.predict(imgs, device=0, conf=0.001, save=True,
                  project='C:/Users/28687/Desktop/X/yolo_train/runs/predict',
                  name='predict_conf001', exist_ok=True, verbose=False)
for x in r:
    name = x.path.replace('\\', '/').split('/')[-1]
    print(name, 'boxes:', len(x.boxes))
    for b in x.boxes:
        print('   cls=%s conf=%.4f xyxy=%s' % (
            model.names[int(b.cls)], float(b.conf),
            [round(float(v), 1) for v in b.xyxy[0]]))

print()
print('=== 2) predict 实际生效的参数 ===')
args = model.predictor.args
for k in ('imgsz', 'conf', 'iou', 'half', 'augment', 'agnostic_nms', 'max_det', 'rect', 'batch', 'mode'):
    print('  %s = %s' % (k, getattr(args, k, '<not set>')))

print()
print('=== 3) val 流水线跑 test split ===')
m = model.val(data=DATA, split='test', device=0, verbose=False)
print('test mAP50:', round(float(m.box.map50), 4), ' mAP50-95:', round(float(m.box.map), 4))
