import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import derivnets


import numpy as np

use_adjoint = True

run_time = 10.0
data_size = 100

true_x0 = torch.Tensor([[1.5],[-0.0]])
t = torch.linspace(0.0, run_time, data_size)

if use_adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class MassSpringDamper(nn.Module):
    def __init__(self, m=1.0, k=1.0, b=0.5):
        super(MassSpringDamper, self).__init__()
        self.m = m
        self.b = b
        self.k = k
        self.A = torch.Tensor([[0, 1],[-self.k/self.m, -self.b/self.m]])

    def forward(self, t, x):
        dx = self.A.mm(x)
        return dx

class PHS_Func(nn.Module):
    def __init__(self):
        super(PHS_Func, self).__init__()
        self.Hnet = derivnets.DerivNet(nn.Linear(2,50), nn.Tanh(), nn.Linear(50,1))
        # self.dnet = nn.Sequential(nn.Linear(1,50), nn.Tanh(), nn.Linear(50,1))
        self.dnet = nn.Sequential(nn.Linear(1, 25), nn.Tanh(), nn.Linear(25, 1))

    def forward(self, t, x):
        H, dHdx = self.Hnet(x.t())
        sd = self.dnet(x[1])
        # d = self.dnet(x[1].abs())
        dx = torch.empty(2, 1)
        dx[0] = dHdx[1]  # q dot
        dx[1] = -dHdx[0] - sd * sd * dHdx[1]
        # dx[1] = -dHdx[0] - sd * sd * x[1]
        # dx[1] = -dHdx[0] - x[1] * d
        return dx


with torch.no_grad():
    true_x = odeint(MassSpringDamper(), true_x0, t, method='dopri5')

model2 = PHS_Func()
model2.load_state_dict(torch.load('./msd_phs6.pt'))
model2.eval()



with torch.no_grad():
    pred_x = odeint(model2, true_x0, t, method='dopri5')


with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 6))
    ax.plot(t.numpy(),true_x[:, 0, 0].numpy())
    ax.plot(t.numpy(), true_x[:, 1, 0].numpy())
    ax.plot(t.numpy(),pred_x[:, 0, 0].detach().numpy(),'-.')
    ax.plot(t.numpy(), pred_x[:, 1, 0].detach().numpy(),'-.')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('states (x)')
    ax.legend(['pos true','vel true','pos model', 'vel model'])
    plt.show()