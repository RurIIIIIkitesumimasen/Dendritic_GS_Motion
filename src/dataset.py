import os
import random
import pickle
import numpy as np
from tqdm import tqdm

import wandb
from wandb import log_artifact

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations

def padding4EfN(x):
    n, i,  h, w = x.shape
    padded_x = np.zeros((n, 3, h, w))#画像size变化
    padded_x[:, :i, :h, :w] = x
  #print(x.shape)
  #x = x.squeeze(1).permute(0, 3, 1, 2)
    return padded_x

class SetData():
    def __init__(self, object_array, root, img_size, seed, noise, noise_num):
        self.train_data, self.train_label, self.valid_data, self.valid_label = self.import_dataset(
            object_array, root, img_size, seed, noise, noise_num)

    def import_dataset(self, object_array, root, img_size, seed, noise, noise_num):

        train_data = []
        train_label = []
        test_data = []
        test_label = []

        train_i = 0
        test_i = 0

        if noise == True:
            root = root + "/" + noise_num

        for load_object in object_array:
            x = np.load(
                'B:/dataset/Motion_GS_dataset_uint8/mixed/image/' + str(load_object) + '.npy'
            #"dataset/MD/GSmotion/no_noise/bg0-1_ob0-1/Object_" + 
            #            str(load_object) + "pixel_32x32_data.npy"
                        )
            t = np.load(
                'B:/dataset/Motion_GS_dataset_uint8/mixed/label/' + str(load_object) + '.npy'
            #"dataset/MD/GSmotion/no_noise/bg0-1_ob0-1/Object_" +
            #            str(load_object) + "pixel_32x32_label.npy"
                        )


            perm = np.random.RandomState(seed=seed).permutation(len(x))
            x = x[perm]
            #x = x[0:10000]
            #print(x.shape)
            #x1 = x1[perm]
            t = t[perm]
            #t = t[0:10000]
           
            

            # Training 各ピクセル--ずつ取り出し
            train_data[train_i:train_i+1800] = x[:1800] * 1
            train_label[train_i:train_i+1800] = t[:1800] * 1
            train_i += 1800

            # Test 各ピクセル--ずつ取り出し
            test_data[test_i:test_i+200] = x[1800:2000] * 1
            test_label[test_i:test_i+200] = t[1800:2000] * 1
            test_i +=200



        train_data = np.array(train_data)
        train_label = np.array(train_label)

        test_data = np.array(test_data)
        test_label = np.array(test_label)


        perm_train = np.random.RandomState(seed=seed).permutation(len(train_data))
        train_data = train_data[perm_train][:7500]

        train_label = train_label[perm_train][:7500]
        train_label = np.argmax(train_label, axis =1)#4ResNet
        perm_test = np.random.RandomState(seed=seed).permutation(len(test_data))
        test_data = test_data[perm_test][:2500]
        test_label = test_label[perm_test][:2500]
        test_label = np.argmax(test_label, axis =1)#4ResNet
        
        # train_data = np.expand_dims(train_data, axis=1)   #4BinaryOri
        train_data = padding4EfN(train_data) #4DeepModel
        # test_data = np.expand_dims(test_data, axis=1)   #4BinaryOri
        test_data = padding4EfN(test_data) #4DeepModel

        return train_data, train_label, test_data, test_label

    def set_train_data_Loader(self, batch_size):
        train_dataset = MotionDetectionDataset(
            data=self.train_data, label=self.train_label)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )

        return train_loader

    def set_valid_data_Loader(self, batch_size):
        valid_dataset = MotionDetectionDataset(
            data=self.valid_data, label=self.valid_label)

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )

        return valid_loader


class MotionDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx].to(torch.float32)
        out_label = self.label[idx]

        # channelを増やす
        # return out_data[np.newaxis, :, :], out_label
        return out_data, out_label #EfN用


