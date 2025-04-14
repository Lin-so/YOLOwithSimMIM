from engine.model import YOLOwithSimMIMModel
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
# 训练器

class Trainer(DetectionTrainer):
    """
    基于YOLODetectionTrainer的训练器,重写了DataSet和Model的初定义
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.pretrain = cfg.get('pretrain',False)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOwithSimMIMModel(cfg, pretrain=self.pretrain)
        if weights:
            model.load(weights)
        return model
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """Swin Transformer对图像大小要求严格,将rect设为False,避免测试报错"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        if self.pretrain:
            self.loss_names = ["l1_loss"]
            if loss_items is not None:
                return {f"{prefix}/{self.loss_names[0]}":loss_items}
            else:
                return [f"{prefix}/{self.loss_names[0]}"]
            
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys
        
    def validate(self):
        if self.pretrain:
            return self.metrics, self.fitness
        return super().validate()
    
    def final_eval(self):
        if self.pretrain:
            return self.metrics
        return super().final_eval()