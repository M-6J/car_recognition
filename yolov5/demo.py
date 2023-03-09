import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import letterbox, non_max_suppression, scale_boxes


class Yolov5Detect(object): #iou : Non-maximum suppression (NMS) 중복 제거 임계값 (default=0.5)
    def __init__(self, weights='./weights/yolov5s.pt', device=0, img_size=(352,352), conf=0.5, iou=0.5):
        with torch.no_grad():
            self.device = "cuda:%s" % device
            self.model = attempt_load(weights, device=self.device) # load FP32 model
            self.model.half() # to FP16
            self.imgsz = img_size  # img_size最好是32的整数倍
            self.conf = conf
            self.iou = iou
            temp_img = torch.zeros((1, 3, self.imgsz[0], self.imgsz[1]), device=self.device)  # init img
            _ = self.model(temp_img.half())  # run once

    def pre_process(self, img_path):
        img0 = cv2.imread(img_path)
        assert img0 is not None, "Image Not Found " + img_path
        img = letterbox(img0, new_shape=self.imgsz,auto=False)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0

    def predict(self, img_path):
        img, img0 = self.pre_process(img_path)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # uint8 to fp16
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]
        return pred, img, img0


    def post_process(self, img_path):
        #
        pred, img, img0 = self.predict(img_path)

        # Apply NMS
        #NMS(non-maximum suppression) 알고리즘을 사용하여 겹치는 박스 중 가장 가능성이 높은 박스를 선택합니다.
        pred = non_max_suppression(pred, self.conf, self.iou, classes=None, agnostic=False)
        pred, im0 = pred[0], img0
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], im0.shape).round()
            pred = pred.cpu().detach().numpy().tolist() # from tensor to list
        return pred, img0


from PIL import Image,ImageDraw,ImageFont
def draw_box_string(img, box, string):
    x,y,w,h = box
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    #font = ImageFont.truetype("simhei.ttf", 24, encoding="utf-8")
    font = ImageFont.truetype("Arial.ttf", 12)
    draw.text((x+w, y), string, (0, 255, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


import os
def get_image_list(image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG','png']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist



if __name__ == '__main__':
    from tqdm import tqdm
    detector = Yolov5Detect()
    img_list = get_image_list("test_imgs")
    for img_path in tqdm(img_list):
        img_name = os.path.basename(img_path)
        pred, img0 = detector.post_process(img_path)
        pred = [i for i in pred if i[-1]==2.0]  # 2.0是car的标签
        if pred is None:
            cv2.imwrite(os.path.join("output", os.path.basename(img_path)), img0)
            continue
        for obj in pred:
            x1, y1, x2, y2, conf, label = obj
            box = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            string = "%s:%.3f" % ("car", conf)
            img0 = draw_box_string(img0, box, string)
        cv2.imwrite(os.path.join("output", img_name), img0)

