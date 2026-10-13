import time
import torch
from ultralytics import YOLO

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print('vram: %.1f GB' % (props.total_memory / 1024**3))
print('compute capability: sm_%d%d' % (props.major, props.minor))

WEIGHTS = 'C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning-2/weights/best.pt'
IMGS = [
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/test/6.jpg',
    'C:/Users/28687/Desktop/X/yolo_train/dataset/images/val/1.jpg',
]

model = YOLO(WEIGHTS)

t0 = time.time()
results = model.predict(IMGS, device=0, save=True,
                        project='C:/Users/28687/Desktop/X/yolo_train/runs/predict',
                        name='gpu_test', exist_ok=True)
print('cold inference (含模型加载+2张图): %.2fs' % (time.time() - t0))

times = []
for _ in range(5):
    t = time.time()
    model.predict(IMGS[0], device=0, verbose=False)
    times.append(time.time() - t)
print('warm inference avg: %.0f ms/张' % (sum(times[1:]) / (len(times) - 1) * 1000))

for r in results:
    dets = [(model.names[int(b.cls)], round(float(b.conf), 2)) for b in r.boxes]
    print('检测:', r.path.split('/')[-1], '->', dets if dets else '无目标')
print('结果图保存在:', results[0].save_dir)
