from ultralytics import YOLO

WEIGHTS = 'C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning-2/weights/best.pt'
imgs = [
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/test/6.jpg',
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/val/1.jpg',
]

model = YOLO(WEIGHTS)
print('classes:', model.names)


def report(tag, results):
    print('---', tag, '---')
    for x in results:
        name = x.path.replace('\\', '/').split('/')[-1]
        n = len(x.boxes)
        if n:
            confs = sorted([round(float(b.conf), 3) for b in x.boxes], reverse=True)
            cls = [model.names[int(c)] for c in x.boxes.cls]
            print(name, '->', n, 'boxes, top conf:', confs[:5], cls[:5])
        else:
            print(name, '-> 无检测')


report('GPU fp16 conf=0.001', model.predict(imgs, device=0, conf=0.001, verbose=False))
report('GPU fp32 conf=0.05', model.predict(imgs, device=0, half=False, conf=0.05, verbose=False))
report('CPU fp32 conf=0.05', model.predict(imgs, device='cpu', conf=0.05, verbose=False))
