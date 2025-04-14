from engine.model import YOLOwithSimMIMModel
from ultralytics.engine.model import Model
from ultralytics.models.yolo.detect import DetectionValidator,DetectionPredictor
from engine.trainer import Trainer

class YOLOwithSimMIM(Model):
    @property
    def task_map(self):
        return {
            "detect": {
                "model": YOLOwithSimMIMModel,
                "trainer": Trainer,
                "validator": DetectionValidator,
                "predictor": DetectionPredictor,
            },
        }