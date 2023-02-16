# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:39:13 2022

@author: unknown
"""

import sys
import torch 
import numpy as np
import random
from captum.attr import Saliency, LayerGradCam, IntegratedGradients, LRP, NoiseTunnel, KernelShap, DeepLift, DeepLiftShap, Occlusion


def input_grads(outputs, x, y):
    return torch.stack([torch.autograd.grad(outputs=out, inputs=x, retain_graph=True, create_graph=True)[0][i] 
                             for i, out in enumerate(outputs.gather(1, y.unsqueeze(1)))])

def integrated_grads(model, x, x_base, y, m = 10, n = 4):
    steps = list(np.linspace(1, m, num=n, dtype=int, axis=0))
    return (x-x_base)*torch.stack([input_grads(model(x_base + k*(x-x_base)/m), x, y) for k in steps]).mean(0)



def choose_attribution(model, method):
    if method == "grad":
        return Saliency(model)
    
    elif method == "intgrad":
        return IntegratedGradients(model)
    
    elif method == "deeplift":
        return DeepLift(model)
    
    elif method == "deepliftshap":
        return DeepLiftShap(model)
    
    elif method == "lrp":
        return LRP(model)
    
    elif method == "shap":
        return KernelShap(model)
    
    elif method == "smoothgrad":
        grad = Saliency(model)
        return NoiseTunnel(grad)
    
    elif method == "occlusion":
        return Occlusion(grad)
    
    elif method == "gradcam":
        return LayerGradCam(model.forward, model.conv1)
        
        
    else:
        sys.exit("Pick an attribution method!")


def apply_roar_mask(model, method, perc, data, target):
    grad = choose_attribution(model, method)
    attr = grad.attribute(data.float().unsqueeze(1), target=target)
    q = [np.percentile(at[at!=0].flatten().numpy(), perc) for at in attr[:,0,:,:]]
    maskpos = torch.ones((attr.shape))
    for i in range(attr.shape[0]):
        maskpos[i,0,attr[i,0,:,:]>q[i]]=0
        data[i,:,:] = data[i,:,:]*maskpos[i,0,:,:]



class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.std  + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class AddSquareMask(object):
    def __init__(self, smin, smax):
        self.smax = smax
        self.smin = smin

    def __call__(self, tensor):
        self.size = random.randint(self.smin, self.smax)
        i = random.randint(self.size, tensor.shape[-1]-self.size)
        j = random.randint(self.size, tensor.shape[-1]-self.size)
        sqr = torch.zeros(tensor.shape)
        sqr[0, i-self.size:i+self.size, j-self.size:j+self.size] = random.uniform(0.5,1)
        tensor[sqr!=0] = sqr[sqr!=0]
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ 


