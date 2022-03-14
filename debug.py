import numpy as np
import torch
from model import MLP
from PDE import BatchBurgers
import matplotlib.pyplot as plt
import math

fig=plt.figure(figsize=(16,12))
grids = [[math.sin(i*2*math.pi/1000-30*j/1000) for i in range(1000)] for j in range(1000)]

outputs = np.array(grids)

ax = fig.add_subplot(111)
im = ax.imshow(outputs)
plt.savefig('tmp.jpg')
plt.show()

