import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov5.demo import Yolov5Detect, draw_box_string
from cars_attributes.MobileNetV2 import mobilenet_v2
from yolov5.utils.general import letterbox, non_max_suppression, box_xywh_to_xyxy
from cars_attributes.test_img import Car_recog

num_classes_color = 15  # 색상 클래스 개수
num_classes_car = 47  # 제조사 클래스 개수
num_classes_type = 9  # 차형 클래스 개수

# YOLOv5 모델과 MobileNetV2 모델의 입력 크기 설정
yolo_input_size = 416  # YOLOv5 모델의 입력 크기
mobilenet_input_size = 224  # MobileNetV2 모델의 입력 크기

#box_xywh_to_xyxy와 yolov5, mobileNet모델 불러오는것.

class YOLOv5_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(YOLOv5_MobileNetV2, self).__init__()
        
        self.num_classes = num_classes
        
        # self.yolo = attempt_load(yolo_weights_path, map_location=torch.device('cpu')).autoshape()  # YOLOv5
        # self.mobilenetv2 = Car_recog(mobilenetv2_weights_path, num_classes=num_classes)  # MobileNetV2
        
        self.yolo = Yolov5Detect("/content/car_recognition/yolov5/weights/yolov5s.pt") # YOLOv5 모델 호출
        self.mobilenet = Car_recog("/content/car_recognition/cars_attributes/checkpoint/mobilenet-v2_20.pth") # MobileNetV2 모델 호출
        
        # # YOLOv5 모델 출력 형식을 MobileNetV2 모델의 입력 형식과 일치시키기 위한 조정
        # num_classes = self.mobilenet.classifier[-1].out_features
        # self.yolo.model[-1].anchor_grid[-1].num_anchors = 3
        # self.yolo.model[-1].predict[-1].predictor[-1].num_classes = num_classes
        # self.yolo.model[-1].predict[-1].predictor[-1].bbox_head[-1].num_classes = num_classes
        
        # YOLOv5 모델과 MobileNetV2 모델의 출력을 결합하는 레이어 정의
        self.combine = nn.Sequential(
            nn.Linear(255, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # YOLOv5로 차량 탐지 수행
        with torch.no_grad():
            detections = self.yolo(x)

        # 추출한 차량 영역을 이용하여 MobileNetV2로 차량 속성 분류 수행
        results = []
        for det in detections:
            if det is not None and len(det):
                det[:, :4] = box_xywh_to_xyxy(det[:, :4])
                det = non_max_suppression(det, 0.4, 0.5, agnostic=True)
                for i, (*xyxy, conf, cls) in enumerate(det):
                    x1, y1, x2, y2 = [int(i) for i in xyxy]

                    # x1, y1, x2, y2 값을 정수로 변환하여 차량 영역 추출
                    cropped_img = x[:, :, y1:y2, x1:x2]

                    # 추출된 이미지를 MobileNetV2 모델의 입력으로 사용하여 차량 속성 분류 수행
                    output = self.mobilenet(cropped_img)

                    # 추출된 속성 분류 결과를 기존 detections에 추가
                    det[i] = torch.cat((det[i][:5], output), dim=1)

                results.append(det)

            else:
                results.append(None)

        # 결합 레이어를 이용하여 YOLOv5과 MobileNetV2의 출력 결합
        combined_results = []
        for det in results:
            if det is not None and len(det):
                combined_results.append(self.combine(det[:, 5:].flatten()))
            else:
                combined_results.append(None)

        return combined_results

    
        
    
# 첫 번째 모델에서는 YOLOv5와 MobileNetV2가 서로 다른 작업(차량 탐지와 차량 분류)을 수행하고, 
# 이를 결합하여 end-to-end 모델을 만듭니다. 
# 반면에, 두 번째 모델에서는 YOLOv5로 탐지한 좌표값을 MobileNetV2에 입력으로 사용하여 
# 차량의 속성을 분류하는 end-to-end 모델을 만듭니다.

# 따라서, 첫 번째 모델과 두 번째 모델은 모두 end-to-end 모델이지만, 
# 첫 번째 모델에서는 YOLOv5와 MobileNetV2가 각각 서로 다른 작업을 수행하고, 
# 이를 결합하여 end-to-end 모델을 만들고, 두 번째 모델에서는 YOLOv5로 탐지한 
# 좌표값을 MobileNetV2에 입력으로 사용하여 end-to-end 모델을 만든다는 차이점이 있습니다.