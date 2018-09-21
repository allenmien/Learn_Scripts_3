# -*-coding:utf-8-*-
"""
@Time   : 2018/9/10 16:08
@Author : Mark
@File   : cnn_torch_script.py
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MINST = True

train_data = torchvision.datasets.MINST(
    root='./minst',
    train=True,
    transform=torchvision.transform.ToTensor(),  # （0, 1），（0， 255）
    download=DOWNLOAD_MINST)

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
