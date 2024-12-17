import os

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
import numpy as np
import time
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_size, tokenizer,dataset_folder,
                 attr,attr_random_rate,mask_folder,training=False,testing=False,valing=False):
        super().__init__()

        self.attr_names = pd.read_csv(attr,header=None,skiprows=1,nrows=1).iloc[0,0].split()
        self.attr = pd.read_csv(attr,header=1,sep='\s+').values

        self.attr_random_rate = attr_random_rate
        self.training = training
        self.testing = testing
        self.valing = valing

        if self.training:
            self.dataset_folder = dataset_folder+"/train"
            self.filename = pd.read_csv(dataset_folder+"/train_list.txt", header=None)[0].tolist()
            self.mask_folder = mask_folder+"/"
        if self.valing:
            self.dataset_folder = dataset_folder+"/val"
            self.filename = pd.read_csv(dataset_folder+"/val_list2000.txt", header=None)[0].tolist()
            self.mask_folder = mask_folder+"/"
        if self.testing:
            self.dataset_folder = dataset_folder + "/train"
            self.filename = pd.read_csv(dataset_folder + "/train_list.txt", header=None)[0].tolist()
            self.mask_folder = mask_folder+"/"

        self.transform = self.build_transform(img_size)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.filename)

    def build_transform(self,img_size):
        transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(img_size),
        ])
        return transform
    def load_mask(self, idx):

        if self.training:
            mask_index = random.randint(0, len(os.listdir(self.mask_folder)) - 1)
            mask = Image.open(self.mask_folder + str(mask_index).zfill(5) + '.png')
        elif self.valing:
            #10-20%:2000 %20-30:4000...
            mask = Image.open(self.mask_folder + str(idx + 2000).zfill(5) + '.png')
        else:
            mask = Image.open(self.mask_folder+str(idx).zfill(6)+'.png')

        return mask

    def __getitem__(self, idx):

        input_img = Image.open(os.path.join(self.dataset_folder, self.filename[idx]))
        # img, (attr, identity, bbox, landmarks) = self.celebahq[0]
        attridx = int(self.filename[idx].split('.')[0])

        if attridx > len(self.attr):
            print("error attridx")
        attr_list = self.attr[attridx]
        attrs = np.array(self.attr_names)[attr_list == 1]
        # 属性标签
        num_to_select = max(1, int(len(attrs) * self.attr_random_rate))
        random_attrs = np.random.choice(attrs, size=num_to_select,replace=False)
        caption = 'This person has '+",".join(random_attrs)

        # input images scaled to 0,1
        gt = self.transform(input_img)
        gt = F.to_tensor(gt)

        mask = self.load_mask(idx)
        if mask.mode == 'RGBA':
            mask = mask.convert('L')
        mask = self.transform(mask)
        mask = F.to_tensor(mask)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        mask = 1-mask
        gt_keep_mask = gt * mask

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "ground_truth": gt,
            "mask": mask,
            "gt_keep_mask":gt_keep_mask,
            "caption": caption,
            "input_ids": input_ids,
            "filename": self.filename[idx]
        }
