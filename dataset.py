from torch.utils.data import Dataset
import PIL.Image as Image
import os
import scipy.io as scio
import torch
import numpy as np


def make_dataset(root):
    imgs = []
    data = os.path.join(root, 'train') 
    label = os.path.join(root, 'label')
    data_file = os.listdir(data)
    label_file = os.listdir(label)

    n = len(os.listdir(label))
    for i in range(n):
        file_name = open('file_name.txt', 'a')
        file_name.write(str(data_file[i]) + '\n')
        file_name.close()
        try:
            img = scio.loadmat(os.path.join(data, data_file[i]))
            mask = scio.loadmat(os.path.join(label, label_file[i]))
        except:
            print(data_file[i])
        a = img['GY_Es']  # 对应电场强度GY_Es
        for j in range(a.shape[0]):
            b = a[j, 0]
            if np.isnan(b):
                print(i)
                print(data_file[i])
        if np.all(img['GY_Es'] == 0):
            print(i)
            loss_data_test = open('error.txt', 'a')
            loss_data_test.write(str(data_file[i]) + '\n')
            loss_data_test.close()
            print(data_file[i])
        # else:
        #     imgs.append((img['input'], mask['label']))
        try:
            imgs.append((img['GY_Es'], mask['GY_sigma']))
        except:
            # loss_data_test = open('error.txt', 'a')
            # loss_data_test.write(str(data_file[i]) + '\n')
            # loss_data_test.close()
            print(data_file[i])
            pass
    return imgs


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data_train, data_label = self.imgs[index]
        data_train_1 = (data_train - data_train.min()) / (data_train.max() - data_train.min())
        data_train = data_train / data_train.max()
        train_mean = np.mean(data_train)
        train_std = np.std(data_train)
        data_train_2 = (data_train - train_mean) / train_std

        # a = np.zeros(shape=(1, 208), dtype='float64')
        # a[0, :] = data_train_1[:, 0]
        a = data_train_2[:, 0]
        data_train = torch.from_numpy(a).float()
        data_label = torch.from_numpy(data_label).squeeze(1).float()
        data_label = data_label

        return data_train, data_label

    def __len__(self):
        return len(self.imgs)
