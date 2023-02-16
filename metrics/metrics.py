# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:41:11 2022

@author: unknown
"""

import numpy as np
import torch
from captum.attr import Saliency
from torch.nn import Softmax 
from utils.utils import choose_attribution
from captum.attr import LayerAttribution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def accuracy(model, loaders):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            pass
    acc = correct/len(loaders["test"])
    return acc


def complexity(model, loaders):
    model.eval()
    grad = Saliency(model)
    for images, labels in loaders['test']:
        images, labels = images.to(device), labels.to(device)
    attr = grad.attribute(images, target=labels)
    normfactor = attr.sum(dim=(2,3))
    pg = attr.squeeze(1).reshape(images.shape[0], images.shape[-1]*images.shape[-1])/normfactor
    mc = -(pg*torch.log(pg+0.01)).sum(1)
    return mc.mean().item()
    

def MoRF(model, loaders, perc, method="grad", model_attr = None):
    model.eval()
    softm = Softmax()
    drops = np.zeros(len(perc))
    if model_attr:
        model_attr.eval()
        grad = choose_attribution(model_attr, method)
    else:
        grad = choose_attribution(model, method)
        
    for images, labels in loaders['test']:
        images, labels = images.to(device), labels.to(device)
        if method == "gradcam":
            attr = grad.attribute(images, target=labels.item())
            attr = LayerAttribution.interpolate(attr, (images.shape[-1],images.shape[-1]) )
            attr = attr.squeeze()
        else:
            attr = grad.attribute(images, target=labels.item())
            attr = attr.squeeze()
        for j in range(len(perc)):
            q3, q1 = np.percentile(attr[attr!=0].flatten().cpu().detach().numpy(), [perc[j], 100-perc[j]])
            maskpos = torch.ones((images.shape[-1], images.shape[-1]))
            maskpos[attr>q3]=0
            d = softm(model(images.cuda()*maskpos.cuda())).cpu().detach().numpy()[0,torch.argmax(softm(model(images.cuda())))]/torch.max(softm(model(images.cuda()))).cpu().detach().numpy()
            drops[j] += d
    
    return drops/len(loaders['test'])


def faithfulness(model, loaders, perc):
    model.eval()
    softm = Softmax()
    grad = Saliency(model)
    corr = []
    for images, labels in loaders['test']:
        images, labels = images.to(device), labels.to(device)
        attr = grad.attribute(images, target=labels.item())
        attr = attr.squeeze()
        sumgrad = np.zeros(len(perc))
        diffpred = np.zeros(len(perc))
        for j in range(len(perc)):
            q3, q1 = np.percentile(attr[attr!=0].flatten().cpu().detach().numpy(), [perc[j], 100-perc[j]])
            maskpos = torch.ones((images.shape[-1], images.shape[-1])).to(device)
            maskpos[attr>q3]=0
            sumgrad[j] = attr[attr>q3].sum()
            diffpred[j] = softm(model(images)).squeeze()[labels].item() - softm(model(maskpos*images)).squeeze()[labels].item()
        
        corr.append(np.corrcoef(sumgrad, diffpred)[0,1])                   
    
    return corr