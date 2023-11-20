import os
import numpy as np
import cv2
import pandas as pd
import random

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import csv
import random 
import face_detection


class NIT_EC(Dataset):
    def __init__(self, anno_path="data/annotations/", transform=None, data_split='train', data_split_name='', image_mode='RGB', advanced=False):
       # self.data_dir = data_dir
        assert data_split == 'train' or data_split == 'test'
        self.data_split = data_split
        self.advanced = advanced
        self.transform = transform
        self.image_mode = image_mode
        self.data_list = []# list(csv.reader(file))

        for filename in os.listdir(anno_path+"train"):
            filename = os.path.join(anno_path,"train",filename)
            file = open(filename, 'r')
            self.data_list = self.data_list + list(csv.reader(file))

        random.Random(2022).shuffle(self.data_list)
        
        # Split data
        ratio = 1 
        if data_split == 'train':
            self.data_list = self.data_list[:round(ratio*len(self.data_list))]      
        elif data_split == 'val':
            self.data_list = self.data_list[round(ratio*len(self.data_list)):]
        else:
            self.data_list = []
            for filename in os.listdir(anno_path+"test"):
                filename = os.path.join(anno_path,"test",filename)
                file = open(filename, 'r')
                self.data_list = self.data_list + list(csv.reader(file))
            

        self.length = len(self.data_list)
        print(f"Dataset count total: {self.length}")


    def __getitem__(self, index):
        # Row:
        # filename,bbox_x1,bbox_y1,bbox_x2,bbox_y2,split,label
        data_row = self.data_list[index]
        base_path = os.getcwd()
        path = os.path.join(base_path,data_row[0])
        img = Image.open(path)
        img = img.convert(self.image_mode)
        width, height = img.size
        x_min,y_min,x_max,y_max = round(float(data_row[1])), round(float(data_row[2])), round(float(data_row[3])), round(float(data_row[4]))
     
     
       # k = 0.2 to 0.40
        if self.data_split == 'train':
            k = np.random.random_sample() * 0.6 + 0.2
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
        
        img_crop = img.crop((max(0,int(x_min)), max(0,int(y_min)), min(width,int(x_max)), min(height,int(y_max))))
        label = int(data_row[5])

        if self.transform is not None:
            img_crop = self.transform(img_crop)
        if label == 1:    
            label = torch.FloatTensor([0, 1])   # Class 0: No eye contact, Class 1: Eye contact
        else:
            label = torch.FloatTensor([1, 0])

        if self.advanced:
            return img_crop, label, path# (bbox_x2-bbox_x1,bbox_y2-bbox_y1)
        else:
            return img_crop, label


    def __len__(self):
        # 122,450
        return self.length
    

