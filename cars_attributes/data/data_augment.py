# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import random
import numpy as np


def gussian_blur(img, blur_prob):
    """
    input: img readed by cv2
           the probability to blur img
    output: blur_img
    """
    if random.random() < blur_prob:
        size = random.randrange(3, 9, 2)
        kernel_size = (size, size)
        sigma = random.uniform(1, 2)
        blur_img = cv2.GaussianBlur(img, kernel_size, sigma)
        return blur_img
    else:
        return img


def gamma_trans(img, trans_prob):
    """
    input: img readed by cv2
           the probality to gamma_trans img
    output: gamma_transformed img
    """
    gamma = random.randint(5, 15) / 10.0
    if random.random() < trans_prob:
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        gamma_img = cv2.LUT(img, gamma_table)
        return gamma_img
    else:
        return img


def distort_image(img, hue, sat, val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')
    hue_c = hsv[:,:,0]
    hue_c += hue
    hue_c[hue_c > 180.] -= 180.
    hue_c[hue_c < 0.] += 180.
    sat_c = hsv[:,:,1]
    sat_c *= sat
    sat_c[sat_c > 255.] = 255.
    val_c = hsv[:,:,2]
    val_c *= val
    val_c[val_c > 255.] = 255.
    # jitter
    bgr = np.uint8(hsv)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_HSV2BGR)
    return bgr

def random_distort_image(img, distort_prob, hue=15, sat=0.75, val=0.75):
    if random.random() < distort_prob:
        dhue = np.random.uniform(-hue, hue)
        dsat = np.random.uniform(sat, 1./sat)
        dval = np.random.uniform(val, 1./val)
        distort_img =  distort_image(img, dhue, dsat, dval) 
        return distort_img
    else:
        return img


def random_wave(img, wave_prob, mu=0, sigma=5):
    if random.random() < wave_prob:
        h, w = img.shape
        img = np.float32(img)
        wave = np.random.normal(mu, sigma, h*w).reshape(h, w)
        img += wave
        img[img > 255.] = 255.
        img[img < 0.] = 0.
        return np.uint8(img)
    else:
        return img


def random_crop(img, crop_prob):
    if random.random() < crop_prob:
        h, w, _ = img.shape
        crop_h = random.randint(int(h*0.9), h-1)
        crop_w = random.randint(int(w*0.9), w-1)
        h_off = random.randint(0, h - crop_h)
        w_off = random.randint(0, w - crop_w)
        return img[h_off:h_off+crop_h, w_off:w_off+crop_w]
    else:
        return img


def show_image(img, name="no name"):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img)


if __name__ == "__main__":
    for i in range(100):
        img = cv2.imread("/home/gp/work/data/images/alg_group/0/depth_20.png", -1)
        #show_image(img)
        #cv2.waitKey(0)
        img = random_crop(img, 1)
        img = random_wave(img, 1)
        #img = random_distort_image(img, 1)
        img = gussian_blur(img, 1)
        img = gamma_trans(img, 1)
        show_image(img)
        cv2.waitKey(0)

