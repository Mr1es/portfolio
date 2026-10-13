import cv2
import glob
from ultralytics import YOLO

print('=== 数据集图片原始尺寸 + 标注数 ===')
for split in ('train', 'val', 'test'):
    for p in sorted(glob.glob('C:/Users/28687/Desktop/X/yolo_train/dataset/images/%s/*' % split)):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        lbl = p.replace('\\', '/').replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        try:
            n = sum(1 for _ in open(lbl))
        except OSError:
            n = '?'
        print('  %-12s %sx%-5d 标注:%s  %s' % (split, w, h, n, p.replace('\\', '/').split('/')[-1]))

print()
print('=== 尺度实验: 同一张图在不同 imgsz 下的预测 ===')
W = 'C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning-2/weights/best.pt'
model = YOLO(W)
for sz in (320, 640, 960, 1280, 1920):
    r = model.predict([
        'C:/Users/28687/Desktop/X/yolo_train/dataset/images/val/1.jpg',
        'C:/Users/28687/Desktop/X/yolo_train/dataset/images/test/6.jpg',
    ], device=0, imgsz=sz, conf=0.05, verbose=False)
    out = []
    for x in r:
        name = x.path.replace('\\', '/').split('/')[-1]
        best = max([float(b.conf) for b in x.boxes], default=0)
        cls = model.names[int(x.boxes.conf.argmax().item())] if len(x.boxes) else '-'
        out.append('%s: max_conf=%.3f (%s)' % (name, best, cls))
    print('imgsz=%4d | %s' % (sz, ' | '.join(out)))
