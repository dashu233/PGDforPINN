from turtle import forward
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,input_dim = 2, hidden_layer = 4, hidden_dim = 32,output_dim=1):
        super().__init__()
        self.layers = []
        for i in range(hidden_layer-1):
            self.layers.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(self.layers)
        self.final_layer = nn.Linear(hidden_dim,output_dim)
        self.act = nn.Tanh()
        
    def forward(self, x):
        for ly in self.layers:
            x = ly(x)
            x = self.act(x)
        x = self.final_layer(x)
        
        return x
    
class CMLP(nn.Module):
    def __init__(self,input_dim = 3, hidden_layer = 4, hidden_dim = 32,output_dim=1):
        super().__init__()
        self.layers = []
        for i in range(hidden_layer-1):
            self.layers.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(self.layers)
        self.final_layer = nn.Linear(hidden_dim,output_dim)
        self.act = nn.Tanh()
        
    def forward(self, x):
        l1 = torch.sin(x[:,0]).unsqueeze(-1)
        l2 = torch.cos(x[:,0]).unsqueeze(-1)
        x = torch.cat([l1,l2,x[:,1].unsqueeze(-1)],dim=-1)
        
        for ly in self.layers:
            x = ly(x)
            x = self.act(x)
        x = self.final_layer(x)
        
        return x
    
class CheatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,1)
        self.layer2 = nn.Linear(1,1)
    def forward(self,x):
        l1 = torch.sin(x[:,0]).unsqueeze(-1)
        l2 = torch.cos(x[:,0]).unsqueeze(-1)
        
        p1 = torch.cos(30*x[:,1]).unsqueeze(-1)
        p2 = torch.sin(30*x[:,1]).unsqueeze(-1)
        
        return self.layer1(p1)*l1 + self.layer2(p2)*l2