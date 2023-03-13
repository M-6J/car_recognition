import torch.nn as nn
import torch.nn.functional as F
from models.yolo import YOLOv5
from models.mobilenetv2 import MobileNetV2

class CarDetector(nn.Module):
    def __init__(self, num_classes):
        super(CarDetector, self).__init__()
        self.backbone = MobileNetV2()
        self.header = YOLOv5(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.header(x)
        return x