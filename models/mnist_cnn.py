# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:12:29 2022

@author: unknown
"""

from torch.nn import Linear, ReLU, Dropout, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Flatten


class CNN3b(Module):
    def __init__(self):
        super(CNN3b, self).__init__()
        self.conv1 = Sequential(         
            Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            ReLU(),                      
            MaxPool2d(kernel_size=2),    
        )
        self.conv2 = Sequential(         
            Conv2d(8, 16, 5, 1, 2),     
            ReLU(),                      
            MaxPool2d(2),                
        )
        self.out = Linear(16 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output


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
        
        self.drop1 = Dropout(0.15)
        
        self.dense1 = Sequential(
            Linear(16384,32)
        )
        
        self.drop2 = Dropout(0.25)
        
        self.dense2 = Sequential(
            Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop1(x)
        x = self.dense1(x)
        x = self.drop2(x)
        x = self.dense2(x)
        return x