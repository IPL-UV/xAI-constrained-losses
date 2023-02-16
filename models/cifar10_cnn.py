# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:12:29 2022

@author: unknown
"""

from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Flatten


class CNN4b(Module):   
    def __init__(self):
        super(CNN4b, self).__init__()

        self.conv1 = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            BatchNorm2d(32),
            ReLU(inplace=True),
        )
        
        self.conv2 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten()
        )
        
        self.dense1 = Sequential(
            Linear(16384, 128)
        )
        
        self.dense2 = Sequential(
            Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x