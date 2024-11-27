'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-26 12:45:38
LastEditTime: 2024-11-27 08:35:27
'''
import os
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
import torch
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def load_mnist(data_path,batch_size):
    #print("loading MNIST")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    batch_size = batch_size
    input_size = 28
    
    #SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    

    transform_train = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    transform_test = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    # 加载train和test的set
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    
    # 加载trainloader与testloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_mnist(data_path,batch_size):
    #print("loading MNIST")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    batch_size = batch_size
    input_size = 28
    
    #SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    

    transform_train = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    transform_test = transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    # 加载train和test的set
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    
    # 加载trainloader与testloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader
    

class CustomDataset(Dataset):
    def __init__(self, data_dir, train, transform=None):
        """
        Args:
            data_dir (str): 图像数据集文件夹路径
            transform (callable, optional): 预处理（数据增强）函数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        # 获取文件夹下所有图像文件
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg') or fname.endswith('.png')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 打开图像文件
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # 如果有transform，则进行处理
        if self.transform:
            image = self.transform(image)
        # if self.train:
        #     image.requires_grad = True
        # else:
        #     image.requires_grad = False
        return image
    

def load_waterdata(data_path,batch_size):
    #print("loading MNIST")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    batch_size = batch_size
    input_size = 28*4
    
    #SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    

    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        #SetRange
    ])
    
    train_dataset = CustomDataset(data_dir=data_path, train=True, transform=transform_train,)
    test_dataset = CustomDataset(data_dir=data_path, train=False, transform=transform_test,)

    # 使用 DataLoader 加载数据
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader