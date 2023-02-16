# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:56:37 2022

@author: unknown
"""

import argparse
import json
from loaders.mnist_loader import MNIST_data, FashionMNIST_data
from torch.optim import Adam, Adadelta
from models.mnist_cnn import CNN3b, CNN4b
import torch
from train import train_xai, train_base
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
    if setup["data"] == "MNIST":
        loaders = MNIST_data(batch_size = setup["batch_size"])
        model = CNN3b().to(device)
        
    elif setup["data"] == "FashionMNIST":
        loaders = FashionMNIST_data(batch_size = setup["batch_size"])
        model = CNN4b().to(device)
    
    else:
        sys.exit("Specify training dataset!")
        
    optimizer = Adam(model.parameters())    
    if setup["penalty"] == "base":
        acc, loss  = train_base(model, loaders, setup["epoch"], optimizer)
    else:
        acc, loss, xloss  = train_xai(model, loaders, setup["epoch"], optimizer, penalty=setup["penalty"], alpha = setup["lambda"])
#    np.save(os.path.join(setup["outfolder"], setup["data"]+"_"+"acc_"+setup["penalty"]+".npy"), acc)
#    np.save(os.path.join(setup["outfolder"], "xloss_"+setup["penalty"]+".npy"), xloss)
    test_acc = accuracy(model,loaders)
    print("Test accuracy = ", test_acc)
    torch.save(model,os.path.join(setup["outfolder"],setup["data"]+"_cnn_"+setup["penalty"]+"_"+str(test_acc)[2:]+".pt"))
