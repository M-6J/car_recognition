import os
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import random
import json
from PIL import Image
sys.path.insert(0, "data/")
#from data_augment import gussian_blur, gamma_trans, random_distort_image, random_wave, random_crop

class Dataset(data.Dataset):
    def __init__(self, img_list, phase='train'):
        with open(img_list, 'r') as fd:
            imgs = json.load(fd)
        self.phase = phase
        random.shuffle(imgs)
        self.imgs = imgs
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index): #image경로, 색상, 차종, 차형
        sample = self.imgs[index]
        img_path = '/content/car_recognition/dataset/cars_test/images/' + sample['image_path']

        # data augment
        data = cv2.imread(img_path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = Image.fromarray(data)
        
        data = data.resize((256, 256))
        data = self.transforms(data)
        label_color = np.int32(sample['color'])
        label_type = np.int32(sample['type'])
        label_car = np.int32(sample['car'])
        print(label_color, label_car, label_type)
        return data.float(), label_color, label_car, label_type

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    
    train_data = Dataset("/content/new_test.json", "train")
    trainloader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    for i, (data, label, label2, label3) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        img *= np.array([0.5, 0.5, 0.5])
        img += np.array([0.5, 0.5, 0.5])
        img *= 255
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        cv2.imshow('img', img)
        if cv2.waitKey(2000):
            continue
    