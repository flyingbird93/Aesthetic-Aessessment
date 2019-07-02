# coding:utf8
import os
from PIL import Image
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

class AVA(data.Dataset):

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

        img_txt = 'ava_rank_siamese.txt'

        df_ava = pd.read_csv(img_txt, sep=' ', header=None)
        df2 = df_ava.sample(frac=rate)

        imgs1 = list(df2[0])
        imgs2 = list(df2[1])
        labels = list(df2[3])

        imgs_num = len(imgs1)

        if self.test:
            self.imgs1 = imgs1[int(0.9 * imgs_num):]
            self.imgs2 = imgs2[int(0.9 * imgs_num):]
            self.labels = labels[int(0.9 * imgs_num):]
        elif train:
            self.imgs1 = imgs1[int(0.6 * imgs_num):]
            self.imgs2 = imgs2[int(0.6 * imgs_num):]
            self.labels = labels[:int(0.6 * imgs_num)]
        else:
            self.imgs1 = imgs1[int(0.6 * imgs_num):int(0.9 * imgs_num)]
            self.imgs2 = imgs2[int(0.6 * imgs_num):int(0.9 * imgs_num)]
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
        img_path1 = self.root + str(self.imgs1[index]) + '.jpg'
        img_path2 = self.root + str(self.imgs2[index]) + '.jpg'

        data1 = Image.open(img_path1)
        data1 = data1.convert('RGB')
        data1 = self.transforms(data1)

        data2 = Image.open(img_path2)
        data2 = data2.convert('RGB')
        data2 = self.transforms(data2)

        label = self.labels[index]

        return data1, data2, label

    def __len__(self):
        return len(self.imgs1)