import torch
import torch.nn.functional as F
from regions import Region 

class AttackPGD():
    def __init__(self, net,step_size, num_steps,region:Region, loss_function):
        super(AttackPGD, self).__init__()
        self.net = net
        self.step_size = step_size
        self.num_steps = num_steps
        self.loss_function = loss_function
        self.region = region
        self.eps = 0.1
        
    def find(self, inputs):
        '''
        in PDE setting, we find the PGD points one by one
        '''
        requires_grads = [x.requires_grad for x in self.net.parameters()]
        self.net.requires_grad_(False)

        x = inputs.detach()
        
        init_noise = torch.zeros_like(x).normal_(0, self.eps / 4)
        x = x + torch.clamp(init_noise, -self.eps / 2, self.eps / 2)
        x = self.region.project(x)
        
        x = x.detach().clone()

        for i in range(self.num_steps):
            #print('xin:{}'.format(x))
            x.requires_grad_()
            logits = self.net(x)
            loss = self.loss_function(logits,x)
            loss.backward()
            x = torch.add(x.detach(),torch.sign(x.grad.detach()), alpha=self.step_size)
            x = torch.min(torch.max(x, inputs - self.eps), inputs + self.eps)
            #print('xmid:{}'.format(x))
            if not self.region.check(x):
                #print('xout:{}'.format(x))
                x = self.region.project(x)
                #print('projected:{}'.format(x))
            x = x.detach().clone()
            # 
        for p, r in zip(self.net.parameters(), requires_grads):
            p.requires_grad_(r)
        return x
