import os
import cv2
import glob
import torch
from torchvision import transforms as T
from torch.nn import DataParallel

from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont
import numpy as np

import sys
sys.path.insert(1, "car_attributes")
sys.path.insert(1, "yolov5")
from cars_attributes.test_img import Car_recog
from yolov5.demo import Yolov5Detect, draw_box_string


color_name = ['blue', 'green', 'purple', 'yellow', 'orange', 'pink', 'red', 'white', 'black', 'brown', 'grey', 'silver', 'gold', 'tan', 'beige']
car_name = [
"AM_General",
"Acura",
"Aston Martin",
"Audi",
"BMW",
"Bentley",
"Bugatti",
"Buick",
"Cadillac",
"Chevrolet",
"Chrysler",
"Daewoo",
"Dodge",
"Eagle_Talon",
"FIAT",
"Ferrari",
"Fisker",
"Ford",
"GMC",
"Geo_Meotro",
"HUMMER",
"Honda",
"Hyundai",
"Infiniti",
"Isuzu",
"Jaguar",
"Jeep",
"Lamborghini",
"Land Rover",
"Lincoln",
"Mini_Cooper",
"Mazda",
"Mclaren",
"Benz",
"Mitsubishi",
"Nissan",
"Plymouth",
"Porsche",
"Rolls-Royce",
"Scion",
"Spyker",
"Suzuki",
"Tesla",
"Toyota",
"Volkswagen",
"Volvo",
"Smart",
]
type_name = ['SUV', 'Sedan', 'Hatchback', 'Convertible', 'Coupe', 'Wagon', 'Truck', 'Van', 'Minivan']

if __name__ == "__main__":
    car_recog = Car_recog("/content/car_recognition/cars_attributes/checkpoint/mobilenet-v2_30.pth")
    detector = Yolov5Detect("yolov5/weights/yolov5s.pt")
    print("load model successfuly")

    img_path = "./dataset/test_imgs/1.jpg"  # 测试图片路径
    pred, img0 = detector.post_process(img_path)
    pred = [i for i in pred if i[-1]==2.0]  # 2.0是汽车的标签
    for obj in pred:
        x1, y1, x2, y2, conf, label = obj
        box = [int(x1), int(y1), int(x2-x1), int(y2-y1)] 
        x1, y1, w, h = box
        img_car = img0[y1:y1+h, x1:x1+w]
        img_RGB, label_color, label_car, label_type = car_recog.recog(img_car)
        result = "Color:%s\nType:%s\nModel:%s" % (color_name[label_color], car_name[label_car], type_name[label_type])
        print("Result:%s" % result)

        img0 = draw_box_string(img0, box, result)
    cv2.imwrite("result.jpg", img0)