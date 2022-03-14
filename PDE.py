import imp
import torch
import torch.nn as nn
import torch.autograd.functional as grad
    
def Burgers(u,x,nu = 0.1):
    # x[0] is x, x[1] is t
    u1, = torch.autograd.grad(u,x,create_graph=True,retain_graph=True)
    u2, = torch.autograd.grad(u1[0],x,create_graph=True,retain_graph=True)
    
    return (u1[1] + u*u1[0] - nu*u2[0])**2


def BatchBurgers(u,x,nu = 0.1):
    # x[0] is x, x[1] is t
    
    # u1 shape is batch x x_dim
    u1, = torch.autograd.grad(torch.sum(u),x,create_graph=True,retain_graph=True)
    
    # u2 shape is batch x 1
    u2, = torch.autograd.grad(torch.sum(u1[:,0]),x,create_graph=True,retain_graph=True)
    
    return nn.MSELoss()(u1[:,1] + u[:,0]*u1[:,0], nu*u2[:,0])


def BatchConvection(u,x,beta=30):
    u1, = torch.autograd.grad(torch.sum(u),x,create_graph=True,retain_graph=True)
    return nn.MSELoss()(u1[:,1],-u1[:,0]*beta)