import torch
from ultralytics import YOLO

W_BEST = 'C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning-2/weights/best.pt'
W_LAST = 'C:/Users/28687/Desktop/X/yolo_train/runs/train/yolotraning-2/weights/last.pt'
DATA = 'C:/Users/28687/Desktop/X/yolo_train/config/dataset.yaml'

print('=== checkpoint 内部状态 ===')
for w in (W_BEST, W_LAST):
    ckpt = torch.load(w, map_location='cpu', weights_only=False)
    print(w.split('/')[-1], '| epoch:', ckpt.get('epoch'),
          '| best_fitness:', ckpt.get('best_fitness'),
          '| date:', ckpt.get('date'),
          '| keys:', sorted(ckpt.keys()))
    tm = ckpt.get('train_metrics')
    if tm:
        print('  train_metrics:', {k: round(float(v), 4) for k, v in list(tm.items())[:8]})

print()
print('=== 用 best.pt 重跑正式 val (split=val) ===')
model = YOLO(W_BEST)
m = model.val(data=DATA, split='val', device=0, verbose=False)
print('mAP50:', round(float(m.box.map50), 4))
print('mAP50-95:', round(float(m.box.map), 4))
print('per-class AP50:', {model.names[int(k)]: round(float(v), 4) for k, v in m.box.ap50_index.items()} if hasattr(m.box, 'ap50_index') else '')
try:
    for i, c in enumerate(m.box.ap_class_index):
        print('  class', model.names[int(c)], 'AP50:', round(float(m.box.ap50[i]), 4))
except Exception as e:
    print('per-class parse fail:', e)
