# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:56:37 2022

@author: unknown
"""

import argparse
import json
from loaders.mnist_loader import MNIST_data, FashionMNIST_data
from torch.optim import Adam
from models.mnist_cnn import CNN3b, CNN4b
import torch
from train import train_base
import os
import numpy as np
from metrics.metrics import accuracy
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
                      
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="path to config file")
args = parser.parse_args()

with open(args.config_path, "r") as jsonfile:
    setup = json.load(jsonfile)
    
if __name__ == '__main__':
    print(setup)
    roar_acc = []
    if setup["data"] == "MNIST":
        loaders = loaders = MNIST_data(batch_size = 30)
    elif setup["data"] == "FashionMNIST":
        loaders = loaders = FashionMNIST_data(batch_size = 30)
    else:
        sys.exit("Choose dataset!")
    tmodel = torch.load(setup["model"])
    tmodel.eval()
    test_acc = accuracy(tmodel,loaders)
    roar_acc.append(test_acc)
    print("Perc removed = ", 0, "Test accuracy = ", test_acc)
    for p in [80, 50, 20, 10]:  
        if setup["data"] == "MNIST":
            loaders = loaders = MNIST_data(batch_size = 30, roar = True, model = tmodel, method = "grad", perc = p)
            model = CNN3b().to(device)
        elif setup["data"] == "FashionMNIST":
            loaders = loaders = FashionMNIST_data(batch_size = 30, roar = True, model = tmodel, method = "grad", perc = p)
            model = CNN4b().to(device)
        else:
            sys.exit("Choose dataset!")
            
        optimizer = Adam(model.parameters()) 
        acc, loss  = train_base(model, loaders, 6, optimizer)
        test_acc = accuracy(model,loaders)
        roar_acc.append(test_acc)
        print("Perc removed = ", 1-p/100, "Test accuracy = ", test_acc)
    
    np.save(os.path.join(setup["outfolder"], setup["data"]+"_"+"roar_"+setup["penalty"]+".npy"), np.array(roar_acc))


