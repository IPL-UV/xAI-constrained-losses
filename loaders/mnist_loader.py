# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:40:16 2022

@author: unknown
"""

import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from utils.utils import AddGaussianNoise, AddSquareMask, apply_roar_mask

device = torch.device('cpu')

def MNIST_data(batch_size = 20, test_batch_size = 1, gauss_noise = False, square_mask = False, mean = 1., std = 1., size_min=2, size_max=6, roar = False, model = None, method = "grad", perc = None):
    
    train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
    
    if gauss_noise:
        test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = transforms.Compose([transforms.ToTensor(),
                                                  AddGaussianNoise(mean, std)])
        )
    elif square_mask:
        test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = transforms.Compose([transforms.ToTensor(),
                                                   AddSquareMask(size_min, size_max)])
        )
    else:
        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )
    
    if roar:
        model.to(device)
        apply_roar_mask(model, method, perc, train_data.data, train_data.targets)
        apply_roar_mask(model, method, perc, test_data.data, test_data.targets)
       

    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    
        'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=test_batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    }
    

    
    return loaders 


def FashionMNIST_data(batch_size = 20, test_batch_size = 1, gauss_noise = False, square_mask = False, mean = 1., std = 1., size_min=2, size_max=6, roar = False, model = None, method = "grad", perc = None):
    
    train_data = datasets.FashionMNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
    
    if gauss_noise:
        test_data = datasets.FashionMNIST(
        root = 'data', 
        train = False, 
        transform = transforms.Compose([transforms.ToTensor(),
                                                  AddGaussianNoise(mean, std)])
        )
    elif square_mask:
        test_data = datasets.FashionMNIST(
        root = 'data', 
        train = False, 
        transform = transforms.Compose([transforms.ToTensor(),
                                                   AddSquareMask(size_min, size_max)])
        )
    else:
        test_data = datasets.FashionMNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )
    
    if roar:
        model.to(device)
        apply_roar_mask(model, method, perc, train_data.data, train_data.targets)
        apply_roar_mask(model, method, perc, test_data.data, test_data.targets)
       

    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    
        'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=test_batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    }
    

    
    return loaders 


