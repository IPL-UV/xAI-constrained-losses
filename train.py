# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:25:02 2022

@author: unknown
"""

import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import input_grads, integrated_grads
from losses.losses import StandardCrossEntropy, FidelityConstraint, SmoothnessConstraint, LocalityConstraint, GradientRegularization, ConsistencyConstraint, SymmetryConstraint, GradReg, Symm, Cons, Smooth, Fid, Loc 
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def train_base(model, loaders, num_epochs, optimizer):
    model.train()
    total_step = len(loaders['train'])
    acc_x_epoch = []
    loss_x_batch = [] 
    loss_func = StandardCrossEntropy()
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images)    # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss_x_batch.append(loss)
            flat_out = np.argmax(output.detach().cpu().numpy(), axis=1)
            correct += (flat_out == b_y.detach().cpu().numpy()).sum()
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            
            if (i + 1) == total_step:
                accuracy =  correct / (total_step*loaders['train'].batch_size)
                print('Accuracy = ', accuracy)
        
        acc_x_epoch.append(accuracy)
    
    return (acc_x_epoch, loss_x_batch)



def train_xai(model, loaders, num_epochs, optimizer, penalty = "gradreg", attribution = "gradients", alpha = 0.1):
    constraints = {"gradreg": GradientRegularization(cweight = alpha), 
               "consistency": ConsistencyConstraint(cweight = alpha), 
               "smoothness": SmoothnessConstraint(cweight = alpha), 
               "fidelity": FidelityConstraint(cweight = alpha), 
               "locality": LocalityConstraint(cweight = alpha), 
               "symmetry": SymmetryConstraint(cweight=alpha)}
    model.train()
    total_step = len(loaders['train'])
    acc_x_epoch = []
    loss_x_batch = []
    xloss = []
    loss_func = constraints[penalty]
    for epoch in range(num_epochs):
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images, requires_grad = True)    # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            grads = input_grads(output, b_x, b_y)
            loss = loss_func(output, grads, b_x, model, b_y, xloss)
#            loss_x_batch.append(loss.detach().cpu().numpy())
            flat_out = np.argmax(output.detach().cpu().numpy(), axis=1)
            correct += (flat_out == b_y.detach().cpu().numpy()).sum()
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()     
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            
            if (i + 1) == total_step:
                accuracy =  correct / (total_step*loaders['train'].batch_size)
                print('Accuracy = ', accuracy)
                
#            b_x = b_x.detach()
#            grads = grads.detach()
#            loss = loss.detach()
#            output = output.detach()
#            del output
#            del loss
#            del b_x
#            del grads
        
        acc_x_epoch.append(accuracy)
    
    return (acc_x_epoch, loss_x_batch, xloss)




def train_savingloss(model, loaders, num_epochs, optimizer):
    model.train()
    total_step = len(loaders['train'])
    acc_x_epoch = []
    loss_x_batch = []
    constraints = {"gradreg": [], "cons": [], "smooth": [], "fid": [], "loc": [], "sym": []}
    loss_func = StandardCrossEntropy()
    grad_reg = GradReg()
    consistency = Cons()
    smoothness = Smooth()
    fidelity = Fid() 
    locality = Loc()
    symmetry = Symm()
    
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images, requires_grad = True)    # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            grads = input_grads(output, b_x, b_y)
            loss = loss_func(output, b_y)
            with torch.no_grad():
                constraints["gradreg"].append(grad_reg(output, grads, b_x, model, b_y).detach().cpu().item())
                constraints["cons"].append(consistency(output, grads, b_x, model, b_y).detach().cpu().item())
                constraints["smooth"].append(smoothness(output, grads, b_x, model, b_y).detach().cpu().item())
                constraints["fid"].append(fidelity(output, grads, b_x, model, b_y).detach().cpu().item())
                constraints["loc"].append(locality(output, grads, b_x, model, b_y).detach().cpu().item())
            symloss = symmetry(output, grads, b_x, model, b_y)
            constraints["sym"].append(symloss.item())
            loss_x_batch.append(loss)
            flat_out = np.argmax(output.detach().cpu().numpy(), axis=1)
            correct += (flat_out == b_y.detach().cpu().numpy()).sum()
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            
            if (i + 1) == total_step:
                accuracy =  correct / (total_step*loaders['train'].batch_size)
                print('Accuracy = ', accuracy)
            
            b_x = b_x.detach()
            grads = grads.detach()
            loss = loss.detach()
            output = output.detach()
            symloss = symloss.detach()
            del output
            del loss
            del b_x
            del grads
            del symloss
            
        
        acc_x_epoch.append(accuracy)
    
    with open('.\results\baseline_constraints.json', 'w') as fp:
        json.dump(constraints, fp)
    
    return (acc_x_epoch, loss_x_batch)


 
