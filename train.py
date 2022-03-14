import imp
from model import MLP,CMLP,CheatModel
from PDE import Burgers,BatchBurgers,BatchConvection
import regions
import torch
import torch.nn as nn
import numpy as np
import math
import PGD
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import os

from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg') 
domain = {
    'Boundary':[
        [regions.YLine(0,1,0),500],
        [regions.YLine(0,1,2*math.pi),500]
    ],
    'Initial':[
        [regions.XLine(0,2*math.pi,0),500],
    ],
    'PDE':[
        [regions.Rectangle(0,2*math.pi,0,1),1500]
    ]
}

def init_function(x):
    # x shape: batch*2
    return torch.sin(x[:,0])

def boundary_function(x):
    # x shape: batch*2
    return torch.zeros_like(x[:,0])


def cirx(x):
    p1 = torch.where(x[:,0]<0.1)
    x[p1][:,0] = 2*math.pi
    p2 = torch.where(x[:,0]>0.1)
    x[p2][:,0] = 0
    return x
    
def circle_function(x,model):
    x = cirx(x)
    return model(x)

def train(model:nn.Module,domain,iters = 100, new_points='constant',output_dir = 'debug'):
    # sample the initial points from given regions
    if not os.path.exists('output'):
        os.makedirs('output')
    output_dir = os.path.join('output',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir,'samples')):
        os.makedirs(os.path.join(output_dir,'samples'))
    if not os.path.exists(os.path.join(output_dir,'heat_map')):
        os.makedirs(os.path.join(output_dir,'heat_map'))
    # if not os.path.exists(os.path.join(output_dir,'025')):
    #     os.makedirs(os.path.join(output_dir,'025'))
    # if not os.path.exists(os.path.join(output_dir,'050')):
    #     os.makedirs(os.path.join(output_dir,'050'))
    # if not os.path.exists(os.path.join(output_dir,'075')):
    #     os.makedirs(os.path.join(output_dir,'075'))
    
    bd_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Boundary']])).cuda()
    init_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Initial']])).cuda()
    pde_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['PDE']])).cuda()
    
    bd_penalty = 1.0
    init_penalty = 1.0
    pde_penalty = 0.5
    
    beta = 30
    
    class WrapConvection():
        def __init__(self) -> None:
            pass
        def __call__(self,u,x):
            return BatchConvection(u,x,beta)
    
    pgd = PGD.AttackPGD(model,0.01,1,domain['PDE'][0][0],WrapConvection())
    optimizer = torch.optim.Adam(model.parameters(),0.001)
    
    writer = SummaryWriter(output_dir)
    
    for it in range(iters):
        if new_points=='constant':
            pass
        elif new_points == 'random':
            bd_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Boundary']])).cuda()
            init_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Initial']])).cuda()
            pde_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['PDE']])).cuda()
        elif new_points == 'pgd':
            pde_points = pgd.find(pde_points).detach().clone()
            bd_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Boundary']])).cuda()
            init_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Initial']])).cuda()
        elif new_points == 'one_step_pgd':
            bd_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Boundary']])).cuda()
            init_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Initial']])).cuda()
            pde_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['PDE']])).cuda()
            pde_points = pgd.find(pde_points).detach().clone()
        elif new_points == 'mixup':
            bd_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Boundary']])).cuda()
            init_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Initial']])).cuda()
            pde_points = pgd.find(pde_points).detach().clone()
            if it%1000 == 0:
                pde_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['PDE']])).cuda()
        else:
            raise NotImplementedError
        # boundary
        
            
        if it%100 == 0:
            fig=plt.figure(figsize=(16,12))
            data = pde_points.detach().clone().cpu().numpy()
            plt.scatter(data[:,0],data[:,1])
            fig.savefig(os.path.join(output_dir,'samples','{}.jpg'.format(it)))
            plt.close('all')
            
            fig=plt.figure(figsize=(16,12))
            grids = [[[i*2*math.pi/1000,j/1000] for i in range(1000)] for j in range(1000)]
            tensor_grids = torch.Tensor(grids).cuda()
            with torch.no_grad():
                H,W,D = tensor_grids.shape
                tensor_grids = tensor_grids.view(-1,D)
                tensor_output = model(tensor_grids).squeeze()
            outputs = (tensor_output.view(H,W)).cpu().numpy()
            
            ax = fig.add_subplot(111)
            im = ax.imshow(outputs)
            fig.savefig(os.path.join(output_dir,'heat_map','{}.jpg'.format(it)))
            plt.close('all')
            
            # xline = np.array([0.01*i-1 for i in range(200)])
            # tensorx = torch.Tensor(xline).cuda().view(-1,1)
            # tensort1 = torch.ones_like(tensorx)*0.25
            # tensort2 = torch.ones_like(tensorx)*0.5
            # tensort3 = torch.ones_like(tensorx)*0.75
            
            # with torch.no_grad():
            #     tensory1 = model(torch.cat((tensorx,tensort1),1))
            #     tensory2 = model(torch.cat((tensorx,tensort2),1))
            #     tensory3 = model(torch.cat((tensorx,tensort3),1))
            
            # fig=plt.figure(figsize=(16,12))
            # plt.plot(xline,tensory1.cpu().numpy())
            # fig.savefig(os.path.join(output_dir,'025','{}.jpg'.format(it)))
            # plt.close('all')
            
            # fig=plt.figure(figsize=(16,12))
            # plt.plot(xline,tensory2.cpu().numpy())
            # fig.savefig(os.path.join(output_dir,'050','{}.jpg'.format(it)))
            # plt.close('all')
            
            # fig=plt.figure(figsize=(16,12))
            # plt.plot(xline,tensory3.cpu().numpy())
            # fig.savefig(os.path.join(output_dir,'075','{}.jpg'.format(it)))
            # plt.close('all')
            
        pred_bd = model(bd_points)
        tar_bd = circle_function(bd_points,model)
        
        loss_bd = nn.MSELoss()(pred_bd,tar_bd)
        writer.add_scalar('loss_vd',loss_bd.item(),it)
        # init 
        pred_init = model(init_points)
        tar_init = init_function(init_points)
        
        loss_init = nn.MSELoss()(pred_init.squeeze(),tar_init)
        writer.add_scalar('loss_init',loss_init.item(),it)
        # PDE loss
        pde_points.requires_grad_(True)
        
        u = model(pde_points)
        loss_pde = BatchConvection(u,pde_points,beta)
        
        writer.add_scalar('loss_pde',loss_pde.item(),it)
        
        loss = loss_bd * bd_penalty + loss_init * init_penalty + loss_pde * pde_penalty
        if it%100 == 0:
            print('iteration:{}'.format(it))
            #print('model weight:{}'.format(model.layer1.weight))
            print('boundary:{},init:{},pde:{}'.format(loss_bd.item(),loss_init.item(),loss_pde.item()))
            print('loss:{}'.format(loss.item()))
        #print('loss:{}'.format(loss.item()))
        writer.add_scalar('loss',loss.item(),it)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pde_points = pde_points.detach().clone()
        # find new points for next iteration

            
            
            
        # else:
        #     pde_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['PDE']])).cuda()
            
        # bd_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Boundary']])).cuda()
        # init_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]) for rg_list in domain['Initial']])).cuda()

        if it%100 == 0:
            
            #print('loss:{}'.format(loss.item()))
            eval_bd_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]*4) for rg_list in domain['Boundary']])).cuda()
            eval_init_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]*4) for rg_list in domain['Initial']])).cuda()
            eval_pde_points = torch.Tensor(np.concatenate([rg_list[0].sample(rg_list[1]*4) for rg_list in domain['PDE']])).cuda()
            
            pred_bd = model(eval_bd_points)
            tar_bd = circle_function(eval_bd_points,model)
            
            loss_bd = nn.MSELoss()(pred_bd,tar_bd)
            
            # init 
            pred_init = model(eval_init_points)
            tar_init = init_function(eval_init_points)
            
            loss_init = nn.MSELoss()(pred_init.squeeze(),tar_init)
            
            # PDE loss
            eval_pde_points.requires_grad_(True)
            loss_pde = 0
            u = model(eval_pde_points)
            loss_pde = BatchConvection(u,eval_pde_points,beta)
            loss = loss_bd * bd_penalty + loss_init * init_penalty + loss_pde * pde_penalty
            print('eval boundary:{},init:{},pde:{}'.format(loss_bd.item(),loss_init.item(),loss_pde.item()))
            print('eval loss:{}'.format(loss.item()))
            writer.add_scalar('loss_eval',loss.item(),it)
    torch.save({'state_dict':model.state_dict()},os.path.join(output_dir,'last_model.pth.tar'))
if __name__ == '__main__':
    model = MLP().cuda()
    train(model,domain,20000,'one_step_pgd','convection_one_step_pgd')
    