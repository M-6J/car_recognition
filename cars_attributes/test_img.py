import os
import cv2
import glob
import torch
from torchvision import transforms as T
from torch.nn import DataParallel

from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from MobileNetV2 import mobilenet_v2



class Car_recog(object):
    def __init__(self, model_path="/content/checkpoint/mobilenet-v2_20.pth"):
        self.device = torch.device("cuda")
        self.net = mobilenet_v2().to(self.device)
        self.net = DataParallel(self.net)
        self.weights = model_path

        self.net.load_state_dict(torch.load(self.weights))

        normalize = T.Normalize(mean = [0.5, 0.5, 0.5],
                                std = [0.5, 0.5, 0.5]
        )
        self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
        ])

    def recog(self, img):
        # img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_RGB = img.copy()
        img = img.resize((256, 256))
        img = self.transforms(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            self.net.eval()
            img_input = img.to(self.device)
            outputs_color, outputs_type, outputs_sub_type = self.net(img_input)
            outputs_color = torch.softmax(outputs_color, 1)
            outputs_type = torch.softmax(outputs_type, 1)
            outputs_sub_type = torch.softmax(outputs_sub_type, 1)

            label_color = outputs_color.argmax()
            label_type = outputs_type.argmax()
            label_sub_type = outputs_sub_type.argmax()
        return img_RGB, label_color, label_type, label_sub_type



if __name__ == "__main__":
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

    car_recog = Car_recog()
    img_list = [os.path.join("./car_recognition/cars_attributes/test_imgs", i) for i in os.listdir("./car_recognition/cars_attributes/test_imgs")]
    for img_path in img_list:
        img = cv2.imread(img_path)
        img_RGB, label_color, label_car, label_type = car_recog.recog(img)
        print(img_path)
        result = "Color:%s, Car:%s, Type:%s" % (color_name[label_color], car_name[label_car], type_name[label_type])
        print("车辆属性识别结果:%s" % result)
        # 把车属性的识别结果画到图上
        draw = ImageDraw.Draw(img_RGB)
        #font = ImageFont.truetype("./simhei.ttf", 24, encoding="utf-8")
        font = ImageFont.load_default()
        draw.text((0, 0), result, (255, 0, 0), font=font)
        img_BGR = cv2.cvtColor(np.array(img_RGB), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join("/content/car_recognition/cars_attributes/result_test", os.path.basename(img_path)), img_BGR)

