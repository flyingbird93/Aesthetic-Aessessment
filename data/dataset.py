# coding:utf8
import os
from PIL import Image
from PIL import ImageFile
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pandas as pd
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

#import piexif

# 关闭警告
import warnings
warnings.filterwarnings('ignore')

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        #
        self.test = test
        self.train = train
        self.transforms = transforms
        self.root = root

        rate = 1

        img_txt = 'data/final_ava_list.txt'

        df_ava = pd.read_csv(img_txt, sep=' ', header=None)
        df2 = df_ava.sample(frac=rate)

        imgs = list(df2[0])
        labels = list(df2[1])

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs[int(0.9 * imgs_num):]
            self.labels = labels[int(0.9 * imgs_num):]
        elif train:
            self.imgs = imgs[:int(0.6 * imgs_num)]
            self.labels = labels[:int(0.6 * imgs_num)]
        else:
            self.imgs = imgs[int(0.6 * imgs_num):int(0.9 * imgs_num)]
            self.labels = labels[int(0.6 * imgs_num):int(0.9 * imgs_num)]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    #T.Resize(256),
                    T.Resize(224),
                    #T.RandomReSizedCrop(224),
                    T.CenterCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.root + str(self.imgs[index]) + '.jpg'

        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = self.labels[index]

        return data, label

    def __len__(self):
        return len(self.imgs)