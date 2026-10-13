import yaml
from pathlib import Path
from ultralytics import YOLO

# 项目根目录：用脚本自身位置推导，保证在任意工作目录下运行都能找到文件
BASE = Path(__file__).resolve().parent

MODEL = str(BASE / 'weights' / 'yolo26n.pt')  # 预训练底模（YOLO26n：端到端无 NMS，推理更快）
DATA = str(BASE / 'config' / 'dataset.yaml')  # 数据集配置文件路径

def main():
    # 加载训练参数
    with open(BASE / 'config' / 'train.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    YOLO(MODEL).train(data=DATA, **cfg)


# Windows 下 DataLoader 用 spawn 创建子进程，会重新导入本文件；
# 必须加这个保护，否则每个子进程都会再启动一次训练导致崩溃。
if __name__ == '__main__':
    main()

# 佛祖保佑，永无BUG
r"""
                  _ooOoo_
                 o8888888o
                 88" . "88
                 (| -_- |)
                 O\ = /O
              ____/`---'\____
            .' \\| |// `
           / \\||| : |||// 
          / _||||| -:- |||||- 
          | | \\\ - /// | |
          | \_| ''\---/'' | |
          \ .-\__ `-` ___/-. /
        ___`. .' /--.--\ `. . __
     ."" '< `.___\_<|>_/___.' >'""
    | | : `- \`.;`\ _ /`;.`/ - ` : | |
    \ \ `-. \_ __\ /__ _/ .-` / /
======`-.____`-.___\_____/___.-`____.-'======
                  `=---='
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           佛祖保佑 永无BUG
"""