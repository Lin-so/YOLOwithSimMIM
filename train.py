from ultralytics.utils import DEFAULT_CFG_DICT
from engine.trainer import Trainer

if __name__ == '__main__':
    cfg = DEFAULT_CFG_DICT
    cfg['data'] = './config/datasets/coco128.yaml'
    cfg['project'] = './runs/detect'
    cfg['epochs'] = 1

    # 预训练
    # cfg['model'] = './yolowithsimmim.yaml'
    # cfg['val'] = False
    # cfg['pretrain'] = True
    # 训练
    # cfg['model'] = './runs/detect/train15/weights/best.pt'  # 加载预训练文件
    cfg['model'] = './config/models/yolowithsimmim.yaml'  # 跳过预训练
    cfg['pretrained'] = True
    
    # 执行训练
    trainer = Trainer(cfg)
    trainer.train()
