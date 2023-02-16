# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:27:01 2022

@author: unknown
"""

import torch
from torch.nn import LogSoftmax, Module, Softmax, CosineSimilarity
import torch.nn.functional as F
from utils.utils import input_grads, integrated_grads
import torchvision.transforms as transforms
import random

class StandardCrossEntropy(Module):
    log_softmax = LogSoftmax()

    def __init__(self):
        super().__init__()

    def forward(self, outputs, y):
        log_probabilities = self.log_softmax(outputs)
        return -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
    
 
class GradientRegularization(Module):
    log_softmax = LogSoftmax()

    def __init__(self, cweight = 0.1):
        super().__init__()
        self.alpha = cweight

    def forward(self, outputs, grad, x, model, y, xl):
        log_probabilities = self.log_softmax(outputs)
        xloss = torch.abs(grad).mean()
        xl.append(xloss.detach().cpu().item())
        return -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)+self.alpha*xloss
    
 
class FidelityConstraint(Module):
    log_softmax = LogSoftmax()
    softmax = Softmax()
    cosim = CosineSimilarity(dim=-1)
    
    def __init__(self, cweight = 1., min_dist = 0.1):
        super().__init__()
        self.alpha = cweight
        self.thr = min_dist
    
    def forward(self, outputs, grad, x, model, y, xl):
        gmax = grad.view(x.size(0), 1, -1).max(2).values.view(x.size(0), 1, 1, 1)
        gmin = grad.view(x.size(0), 1, -1).min(2).values.view(x.size(0), 1, 1, 1)
        ngrad = (grad - gmin)/(gmax - gmin)
        x_masked = x*(1-ngrad)
        outputs0 = model(x_masked)
        dist = torch.abs(self.cosim(outputs, outputs0))
        xloss = torch.max(torch.as_tensor(0).cuda(), dist.sum()/y.size(0)-torch.as_tensor(self.thr).cuda())
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
        xl.append(xloss.detach().cpu().item())
        return celoss + self.alpha*xloss
    

class SymmetryConstraint(Module):
    log_softmax = LogSoftmax()
    softmax = Softmax()
    cosim = CosineSimilarity(dim=-1)
    
    def __init__(self, cweight = 1.):
        super().__init__()
        self.alpha = cweight
    
    def forward(self, outputs, grad, x, model, y, xl):
        angle = random.choice([5, 10, 15, 20])
        x_rot =  transforms.functional.rotate(x, angle)
        outputs0 = model(x_rot)
        grad0 = input_grads(outputs0, x_rot, y)
        grad_rot = transforms.functional.rotate(grad, angle)
        grad_rot = grad_rot.squeeze().reshape(grad_rot.shape[0], grad_rot.shape[-1]**2)
        grad0 = grad0.squeeze().reshape(grad0.shape[0], grad0.shape[-1]**2)
        xloss = 1-torch.abs(self.cosim(grad0, grad_rot))
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0) 
        xl.append(xloss.mean().detach().cpu().item())
        return celoss + self.alpha*xloss.mean()
    

class LocalityConstraint(Module):
    log_softmax = LogSoftmax()
    
    def __init__(self, cweight = 1., min_grad = 0.01):
        super().__init__()
        self.alpha = cweight
        self.smt = min_grad
    
    def forward(self, outputs, grad, x, model, y, xl):
        gmax = grad.view(x.size(0), 1, -1).max(2).values.view(x.size(0), 1, 1, 1)
        gmin = grad.view(x.size(0), 1, -1).min(2).values.view(x.size(0), 1, 1, 1)
        ngrad = (grad - gmin)/(gmax - gmin)
        xloss = -(x*torch.log(ngrad+self.smt) + (torch.as_tensor(1.)-x)*torch.log(torch.as_tensor(1.)-ngrad+self.smt)).mean()
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
#        print("loc = ", xloss)
        xl.append(xloss.detach().cpu().item())
        return celoss + self.alpha*xloss
    

class ConsistencyConstraint(Module):
    log_softmax = LogSoftmax()
    softmax = Softmax()
    cosim = CosineSimilarity(dim=-1)
    
    def __init__(self, cweight = 1.):
        super().__init__()
        self.alpha = cweight
    
    def forward(self, outputs, grad, x, model, y, xl):
        xloss = 0
        gmax = grad.view(grad.size(0), 1, -1).max(2).values.view(grad.size(0), 1, 1, 1)
        gmin = grad.view(grad.size(0), 1, -1).min(2).values.view(grad.size(0), 1, 1, 1)
        ngrad = (grad - gmin)/(gmax - gmin)
        for n in range(10):
            nsmpl = int(ngrad[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:].shape[0])
            for i in range(nsmpl):
                for j in range((i+1),nsmpl):
                    xloss += (1-self.cosim( ngrad[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:].reshape(nsmpl, grad.shape[-1]*grad.shape[-1])[i,:],  ngrad[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:].reshape(nsmpl, grad.shape[-1]*grad.shape[-1])[j,:]))/(1-self.cosim(x[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:].reshape(nsmpl, x.shape[-1]*x.shape[-1])[i,:], x[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:].reshape(nsmpl, x.shape[-1]*x.shape[-1])[j,:]))
        xloss /= y.size(0)
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
        xl.append(xloss.detach().cpu().item())
        ngrad = ngrad.detach()
        del ngrad
        return celoss + self.alpha*xloss


class SmoothnessConstraint(Module):
    log_softmax = LogSoftmax()

    def __init__(self, cweight = 0.1):
        super().__init__()
        self.alpha = cweight

    def forward(self, outputs, grad, x, model, y, xl):
        grad = grad.squeeze()
        d = grad.shape[-1]
        log_probabilities = self.log_softmax(outputs)
        xloss = torch.abs(torch.roll(F.pad(grad, (0,0,1,1)), 1, 1)[:,:d,:] - grad) + torch.abs(torch.roll(F.pad(grad, (1,1,0,0)), 1, 2)[:,:,:d]-grad)
        xl.append(xloss.mean().detach().cpu().item())
        return -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)+self.alpha*xloss.mean()
    

class GeneralizabilityConstraint(Module): 
    log_softmax = LogSoftmax()
    softmax = Softmax()
    cosim = CosineSimilarity(dim=-1)
    
    def __init__(self, cweight = 1.):
        super().__init__()
        self.alpha = cweight
    
    def forward(self, outputs, ngrad, model, x, y):
        xloss = 0
        for n in range(10):
            cgrad = ngrad[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:]
            for i in range(cgrad.shape[0]):
                x1 = x[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:]*cgrad[i,:,:,:].unsqueeze(0)
                outputs1 = model(x1)
                log_prob1 = self.log_softmax(outputs1)
                xloss += log_prob1.gather(1, y[(torch.argmax(self.softmax(outputs), dim=1) == n)].unsqueeze(1)).sum()/y[(torch.argmax(self.softmax(outputs), dim=1) == n)].size(0)
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
        return celoss - self.alpha*xloss
 


class GradReg(Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, grad, x, model, y):
        xloss = torch.abs(grad).mean()
        return xloss
    
 
class Fid(Module):
    cosim = CosineSimilarity(dim=-1)
    def __init__(self, min_dist = 0.1):
        super().__init__()
        self.thr = min_dist
    
    def forward(self, outputs, grad, x, model, y):
        gmax = grad.view(x.size(0), 1, -1).max(2).values.view(x.size(0), 1, 1, 1)
        gmin = grad.view(x.size(0), 1, -1).min(2).values.view(x.size(0), 1, 1, 1)
        ngrad = (grad - gmin)/(gmax - gmin)
        x_masked = x*(1-ngrad)
        outputs0 = model(x_masked)
        dist = torch.abs(self.cosim(outputs, outputs0))
        xloss = torch.max(torch.as_tensor(0).cuda(), dist.sum()/y.size(0)-torch.as_tensor(self.thr).cuda())
        return xloss
    

class Symm(Module):
    cosim = CosineSimilarity(dim=-1)
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, grad, x, model, y):
        angle = random.choice([5, 10, 15, 20, 25, 30])
        x_rot =  transforms.functional.rotate(x, angle)
        outputs0 = model(x_rot)
        grad0 = input_grads(outputs0, x_rot, y)
        grad_rot = transforms.functional.rotate(grad, angle)
        grad_rot = grad_rot.squeeze().reshape(grad_rot.shape[0], grad_rot.shape[-1]**2)
        grad0 = grad0.squeeze().reshape(grad0.shape[0], grad0.shape[-1]**2)
        xloss = 1-torch.abs(self.cosim(grad0, grad_rot))
        return xloss.mean()
    

class Loc(Module):  
    def __init__(self, min_grad = 0.01):
        super().__init__()
        self.smt = min_grad
    
    def forward(self, outputs, grad, x, model, y):
        gmax = grad.view(x.size(0), 1, -1).max(2).values.view(x.size(0), 1, 1, 1)
        gmin = grad.view(x.size(0), 1, -1).min(2).values.view(x.size(0), 1, 1, 1)
        ngrad = (grad - gmin)/(gmax - gmin)
        xloss = -(x*torch.log(ngrad+self.smt) + (torch.as_tensor(1.)-x)*torch.log(torch.as_tensor(1.)-ngrad+self.smt)).mean()
        return xloss
    

class Cons(Module):
    cosim = CosineSimilarity(dim=-1)
    softmax = Softmax()
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, grad, x, model, y):
        xloss = 0
        gmax = grad.view(grad.size(0), 1, -1).max(2).values.view(grad.size(0), 1, 1, 1)
        gmin = grad.view(grad.size(0), 1, -1).min(2).values.view(grad.size(0), 1, 1, 1)
        ngrad = (grad - gmin)/(gmax - gmin)
        for n in range(10):
            cgrad = ngrad[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:]
            cgrad = cgrad.reshape(cgrad.shape[0], cgrad.shape[-1]*cgrad.shape[-1])
            cx = x[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:]
            cx = cx.reshape(cgrad.shape[0], x.shape[-1]*x.shape[-1])
            for i in range(cgrad.shape[0]):
                for j in range((i+1),cgrad.shape[0]):
                    xloss += (1-self.cosim(cgrad[i,:], cgrad[j,:]))/(1-self.cosim(cx[i,:], cx[j,:]))
        xloss /= y.size(0)
        return xloss


class Smooth(Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, grad, x, model, y):
        grad = grad.squeeze()
        d = grad.shape[-1]
        xloss = torch.abs(torch.roll(F.pad(grad, (0,0,1,1)), 1, 1)[:,:d,:] - grad) + torch.abs(torch.roll(F.pad(grad, (1,1,0,0)), 1, 2)[:,:,:d]-grad)
        return xloss.mean()
    