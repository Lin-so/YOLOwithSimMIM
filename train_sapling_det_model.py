

"""

    训练sapling识别模型

"""

import os

import torch

from ultralytics import YOLO
from ultralytics import YOLOWorld
from ultralytics import RTDETR
from model import YOLOwithSimMIM

if __name__ == '__main__':
    # Load a model
    # yolov8x.pt
    # yolov9e.pt
    # yolov10x.pt
    # yolov8x-obb.pt
    # yolov8x-worldv2.pt
    # rtdetr-x.pt
    # pre_model_path = 'model_hub_storage/yolo11x-obb.pt'
    pre_model_path = './config/models/yolov11x-simmim.yaml'
    # existed_model_path = r'sapling_det_train/train3/weights/last.pt'

    model_path = pre_model_path
    print(os.path.exists(model_path))  # 应该返回 True

    # model = YOLO(model_path)  # load a pretrained model (recommended for training)
    # model = YOLOWorld(model_path)  # 采用YOLO-World
    # model = RTDETR(model_path)
    model = YOLOwithSimMIM(model_path)

    # Train the model
    results = model.train(#data=r'datasets/sapling_det/data.yaml',
                          data = r'./config/datasets/coco128.yaml',
                          epochs=2000,
                          patience=1000,
                          imgsz=640,  # 640, 840, 960
                          batch=20,  # 48, 16
                          single_cls=False,
                          cache=True,
                          device = ['cpu', ],
                          #device=['0',],
                          verbose=True,
                          seed=42,
                          resume=False,
                          flipud=0.1,
                          mixup=0.1,
                          copy_paste=0.1,
                          bgr=0.1,
                          project='sapling_det_train',
                          )
    #
    # metrics = model.val(data=r'datasets/sapling_det/data_valid.yaml',)
    #
    # print(metrics)

    print('Train is done!')