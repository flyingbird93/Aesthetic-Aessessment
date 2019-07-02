# _*_coding: utf-8 _*_

import numpy as np
import cv2
import random
from tqdm import tqdm

compute_path = 'final_ava_list.txt'

cnum = 255508

root = '/home/flyingbird/Smith/Data/AVA/images/'

img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, std = [], []

with open(compute_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)

    for i in tqdm(range(cnum)):
        img_path = lines[i].rstrip().split(' ')[0]
        img_path = root + img_path + '.jpg'

        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))

        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32)/255

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    std.append(np.std(pixels))

means.reverse()
std.reverse()

print('normMean= {}'.format(means))
print('normStd= {}'.format(std))
print('transforms.Normalize(normMean = {}, normstd = {}'.format(means, std))