import torch
import torch.nn as nn
import os

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from cars_attributes.data.dataload import download_dataset
from cars_attributes.data.dataset import Dataset
from cars_attributes.MobileNetV2 import mobilenet_v2
from yolov5.demo import Yolov5Detect
from cars_attributes.WarmUpLR import WarmUpLR
from endtoend import YOLOv5_MobileNetV2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

train_list = "./car_recognition/cars_annos/xin_train.json"
val_list = "./car_recognition/cars_annos/xin_test.json"

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001
max_epoch = 30

# Save the model
def save_model(model, save_path, name, iter_cnt):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

# Train and test the model
def train(epoch, net, trainloader, optimizer, criterion_detector, criterion_classifier, warmup_scheduler):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_color = 0
    total_color = 0
    batch_id = 0
    for (inputs, targets_detector, targets_car, targets_type, targets_color) in tqdm(trainloader):
        if epoch < 2:
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()
            print("warm_lr:%s" % warm_lr)
        inputs, targets_detector, targets_car, targets_type, targets_color = inputs.to(device), targets_detector.to(device), targets_color.to(device), targets_car.to(device), targets_type.to(device)
        optimizer.zero_grad()
        outputs_detector, outputs_color, outputs_car, outputs_type = net(inputs)
        
        # Calculate detection loss
        loss_detector = criterion_detector(outputs_detector, targets_detector.long())
        
        # Calculate classification loss
        loss_color = criterion_classifier(outputs_color, targets_color.long())
        loss_car = criterion_classifier(outputs_car, targets_car.long())
        loss_type = criterion_classifier(outputs_type, targets_type.long())

        # Combine the detection loss and classification loss
        loss = loss_detector + loss_color + loss_car + loss_type
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
            #tensor_board.visual_loss("train_loss", los, iters)
            #tensor_board.visual_acc("train_acc", acc, iters)
        batch_id += 1

    # Test
def test(epoch, net, valloader, criterion_detector, criterion_classifier):
    net.eval()
    test_loss = 0
    correct_detector = 0
    correct_color = 0
    correct_car = 0
    correct_type = 0
    total_detector = 0
    total_color = 0
    total_car = 0
    total_type = 0
    with torch.no_grad():
         for batch_idx, (inputs, targets_detector, targets_car, targets_type, targets_color) in enumerate(valloader):
            inputs, targets_detector, targets_car, targets_type, targets_color = inputs.to(device), targets_detector.to(device), targets_color.to(device), targets_car.to(device), targets_type.to(device)
            outputs_detector, outputs_color, outputs_car, outputs_type = net(inputs)
            
            loss_detector = criterion_detector(outputs_detector, targets_detector.long())
            loss_color = criterion_classifier(outputs_color, targets_color.long())
            loss_car = criterion_classifier(outputs_car, targets_car.long())
            loss_type = criterion_classifier(outputs_type, targets_type.long())

            test_loss += loss_detector.item()
            test_loss += loss_color.item()
            test_loss += loss_car.item()
            test_loss += loss_type.item()
            
            _, predicted_detector = outputs_detector.max(1)
            _, predicted_color = outputs_color.max(1)
            _, predicted_car = outputs_car.max(1)
            _, predicted_type = outputs_type.max(1)
            
            total_detector += targets_detector.size(0)
            correct_detector += predicted_detector.eq(targets_detector.long()).sum().item()
            
            total_color += targets_color.size(0)
            correct_color += predicted_color.eq(targets_color.long()).sum().item()

            total_car += targets_car.size(0)
            correct_car += predicted_car.eq(targets_car.long()).sum().item()

            total_type += targets_type.size(0)
            correct_type += predicted_type.eq(targets_type.long()).sum().item()

    acc_detector = 1.*correct_detector/total_detector
    acc_color = 1.*correct_color/total_color
    acc_car = 1.*correct_car/total_car
    acc_type = 1.*correct_type/total_type

    print("Acc_detector in the val_dataset:%s" % acc_detector)
    print("Acc_color in the val_dataset:%s" % acc_color)
    print("Acc_car in the val_dataset:%s" % acc_car)
    print("Acc_type in the val_dataset:%s" % acc_type)

    if epoch % 10 == 0:
        save_model(net, "./checkpoint/", "EndToEnd", epoch)
        
if __name__ == "__main__":
    
    # Model and optimizer]
    net = YOLOv5_MobileNetV2().to(device)
    
    dataset_name = "custom-stanford-cars-dataset"
    save_path = "./car_recognition/dataset"
    
    #download_dataset(dataset_name, save_path)
    
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
    # train_dataset = CarDataset("train.txt", transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test_dataset = CarDataset("test.txt", transform)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # Loss function
    criterion_detector = nn.BCEWithLogitsLoss()
    criterion_classifier = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    warmup_epoch = 2
    scheduler = CosineAnnealingLR(optimizer, 30 - warmup_epoch)
    
    iter_per_epoch = len(train_dataset)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)


    #tensor_board = TensorBoard(64, 3, 256, 256)
    #tensor_board.visual_model(net)
    for epoch in range(1, max_epoch+1):
        if epoch >= warmup_epoch:
            scheduler.step()
            learn_rate = scheduler.get_lr()[0]
            print("Learn_rate:%s" % learn_rate)
        test(epoch, net, valloader, criterion_detector, criterion_classifier)
        train(epoch, net, trainloader, optimizer, criterion_detector, criterion_classifier, warmup_scheduler)
    
# # Define object detection loss function
# class DetectionLoss(nn.Module):
#     def __init__(self, weight=1.0):
#         super(DetectionLoss, self).__init__()
#         self.weight = weight
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.mse_loss = nn.MSELoss()

#     def forward(self, pred, target):
#         # Calculate objectness loss
#         objectness_loss = self.bce_loss(pred[..., 0], target[..., 0])

#         # Calculate localization loss
#         loc_loss = self.mse_loss(pred[..., 1:5], target[..., 1:5])

#         # Calculate classification loss
#         class_loss = self.bce_loss(pred[..., 5:], target[..., 5:])

#         # Combine the three losses
#         loss = objectness_loss + self.weight * loc_loss + class_loss
#         return loss

# criterion_detector = DetectionLoss()