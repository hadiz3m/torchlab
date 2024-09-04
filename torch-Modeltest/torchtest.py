# 1) designe our model (input , output size , forward pass)
# 2) constract loss and optimizer
# 3) trining loop
#  forward apss
#  backward pass
#  update waights 

import torch
import numpy as np
import torch.nn as nn


X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

text_x = torch.tensor([12],dtype=torch.float32)
n_sample , n_fiture = X.shape

inputsize = n_fiture
outputsize = n_fiture

model = nn.Linear(inputsize,outputsize)

print(f'predict befor training: f(5) = {model(text_x).item():.3f}')

learningrate = 0.1
n_iter = 300
loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr=learningrate)


for epoch in range(n_iter):
    # print('X: ',X)
    y_perd = model(X)

    # print('Y: ',Y,y_perd)
   
    l = loss(Y,y_perd)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 ==0:
        [w,b] = model.parameters()
        print(f' epoch {epoch+1} : w={w[0][0].item():3f} , loss = {l:.8f} ')


print(f'predict after training: f(5) = {model(text_x).item():.3f}')

