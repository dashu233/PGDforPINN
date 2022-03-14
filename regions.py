from tables import UnImplemented


import numpy as np

class Region:
    def __init__(self) -> None:
        pass
    def sample(self,n):
        '''
        sample n points from the area randomly
        '''
        raise UnImplemented
    def check(self,x):
        return True
    def project(self,x):
        print('Warn: Original Projection will do nothing')
        return x


class XLine(Region):
    def __init__(self,x1,x2,y) -> None:
        super().__init__()
        self.left = min(x1,x2)
        self.right = max(x1,x2)
        self.y = y
    
    def sample(self, n):
        x = np.random.rand(n)*(self.right-self.left) + self.left
        y = np.ones_like(x) * self.y        
        return np.transpose(np.array([x,y]))
    def project(self, x):
        L = (x[0].detach()-self.left)//(self.right-self.left)
        x[0] = x[0] - L*(self.right-self.left)
        x[1] = self.y
        return x
    
class YLine(Region):
    '''
    x -> y, y -> x at sample
    '''
    def __init__(self,x1,x2,y) -> None:
        super().__init__()
        self.left = min(x1,x2)
        self.right = max(x1,x2)
        self.y = y
    
    def sample(self, n):
        x = np.random.rand(n)*(self.right-self.left) + self.left
        y = np.ones_like(x) * self.y        
        return np.transpose(np.array([y,x]))
    def project(self, x):
        L = (x[1].detach()-self.left)//(self.right-self.left)
        x[1] = x[1] - L*(self.right-self.left)
        x[0] = self.y
        return x

class Rectangle(Region):
    def __init__(self,x1,x2,y1,y2) -> None:
        super().__init__()
        self.left = min(x1,x2)
        self.right = max(x1,x2)
        self.up = max(y1,y2)
        self.down = min(y1,y2)
    def area(self):
        return (self.up - self.down) * (self.right - self.left)
    
    def sample(self, n):
        x = np.random.rand(n)*(self.right-self.left) + self.left
        y = np.random.rand(n)*(self.up - self.down) + self.down
        
        return np.transpose(np.array([x,y]))
    
    def project(self, x):
        L = ((x[:,0].detach()-self.left)/(self.right-self.left)).floor()
        x[:,0] = x[:,0] - L*(self.right-self.left)
        #print('Lx:{}'.format(L))
        L = ((x[:,1].detach()-self.down)/(self.up-self.down)).floor()
        x[:,1] = x[:,1] - L*(self.up-self.down)
        #print('Ly:{}'.format(L))
        return x
    
    def check(self, x):
        return True