from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torch.utils import data
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

import os
import argparse
import json
from tqdm import tqdm

from data.dataset import Dataset
from tensorboard_gp import TensorBoard
from MobileNetV2 import mobilenet_v2
from WarmUpLR import WarmUpLR

device = 'cuda'
best_acc = 0
max_epoch = 30
#train_list = "/home/guopei/workspace/dataset/traffic/crop_car_imgs/car_imgs/kakou/train.txt"
#val_list = "/home/guopei/workspace/dataset/traffic/crop_car_imgs/car_imgs/kakou/val.txt"



def save_model(model, save_path, name, iter_cnt):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def train(epoch, net, trainloader, optimizer, criterion, warmup_scheduler):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_color = 0
    total_color = 0
    batch_id = 0
    for (inputs, targets_color, targets_car, targets_type) in tqdm(trainloader):
        if epoch < 5:
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()
            print("warm_lr:%s" % warm_lr)
        inputs, targets_color, targets_car, targets_type = inputs.to(device), targets_color.to(device), targets_car.to(device), targets_type.to(device)
        optimizer.zero_grad()
        outputs_color, outputs_car, outputs_type = net(inputs)
        loss_color = criterion(outputs_color, targets_color.long())
        loss_car = criterion(outputs_car, targets_car.long())
        loss_type = criterion(outputs_type, targets_type.long())

        loss = loss_color + loss_car + loss_type
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


        _, predicted_color = outputs_color.max(1)
        total_color += targets_color.size(0)
        correct_color += predicted_color.eq(targets_color.long()).sum().item()

        iters = epoch * len(trainloader) + batch_id
        if iters % 10 == 0:
            acc = predicted_color.eq(targets_color.long()).sum().item()*1.0/targets_color.shape[0]
            los = loss*1.0/targets_color.shape[0]
            tensor_board.visual_loss("train_loss", los, iters)
            tensor_board.visual_acc("train_acc", acc, iters)
        batch_id += 1
        

def test(epoch, net, valloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct_color = 0
    correct_car = 0
    correct_type = 0
    total_color = 0
    total_car = 0
    total_type = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets_color, targets_car, targets_type) in enumerate(valloader):
            inputs, targets_color, targets_car, targets_type = inputs.to(device), targets_color.to(device), targets_car.to(device), targets_type.to(device)
            outputs_color, outputs_car, outputs_type = net(inputs)
            loss_color = criterion(outputs_color, targets_color.long())
            loss_car = criterion(outputs_car, targets_car.long())
            loss_type = criterion(outputs_type, targets_type.long())

            test_loss += loss_color.item()
            test_loss += loss_car.item()
            test_loss += loss_type.item()
            _, predicted_color = outputs_color.max(1)
            _, predicted_car = outputs_car.max(1)
            _, predicted_type = outputs_type.max(1)

            total_color += targets_color.size(0)
            correct_color += predicted_color.eq(targets_color.long()).sum().item()

            total_car += targets_car.size(0)
            correct_car += predicted_car.eq(targets_car.long()).sum().item()

            total_type += targets_type.size(0)
            correct_type += predicted_type.eq(targets_type.long()).sum().item()

    # Save checkpoint.
    acc_color = 1.*correct_color/total_color
    acc_car = 1.*correct_car/total_car
    acc_type = 1.*correct_type/total_type
    tensor_board.visual_acc("test_acc", acc_color, epoch)

    print("Acc_color in the val_dataset:%s" % acc_color)
    print("Acc_car in the val_dataset:%s" % acc_car)
    print("Acc_type in the val_dataset:%s" % acc_type)

    if acc_color > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc_color,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net, './checkpoint/best_model.pth')
        best_acc = acc_color

    if epoch % 10 == 0:
        save_model(net, "./checkpoint/", "mobilenet-v2", epoch)




if __name__ == "__main__":
    # set net
    net = mobilenet_v2(pretrained=True)#훈련 네트워크 설정
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    # data.Dataset() 보니 txt에 경로, 색상, 유형, 차종 이 문자열이 아니라 숫자로 표현되어 있다. 
    # 이유는 숫자가 연산량이 더 적어서 인것같다
    annos_file_path = './dataset/cars_annos/'
    with open(annos_file_path + 'cars_train_annos.json', 'r') as outfile:        
        train_data = json.load(outfile)
    with open(annos_file_path + 'cars_test_annos.json', 'r') as outfile:       
        val_data = json.load(outfile)
    train_list = {}
    val_list = {}
    for i in range(len(train_data)):
        train_list[i] = './dataset/cars_test/images/' + train_data[i]['image_path'] + " " + train_data[i]['color'] + " " + train_data[i]['type'] + " " + train_data[i]['car']
    for i in range(len(val_data)):
        val_list[i] = './dataset/cars_test/images/' + val_data[i]['image_path'] + " " + val_data[i]['color'] + " " + val_data[i]['type'] + " " + val_data[i]['car']

    train_dataset = Dataset(train_list, phase='train')
    val_dataset = Dataset(val_list, phase='val')
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=4)
    valloader = data.DataLoader(val_dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=4)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    warmup_epoch = 5
    scheduler = CosineAnnealingLR(optimizer, 30 - warmup_epoch)

    iter_per_epoch = len(train_dataset)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)


    tensor_board = TensorBoard(64, 3, 256, 256)
    #tensor_board.visual_model(net)
    for epoch in range(1, max_epoch+1):
        if epoch >= warmup_epoch:
            scheduler.step()
            learn_rate = scheduler.get_lr()[0]
            print("Learn_rate:%s" % learn_rate)
        test(epoch, net, valloader, criterion)
        train(epoch, net, trainloader, optimizer, criterion, warmup_scheduler)
