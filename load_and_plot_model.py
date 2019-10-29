import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd


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

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        return self.net(x.t()).t()


with torch.no_grad():
    true_x = odeint(MassSpringDamper(), true_x0, t, method='dopri5')

model2 = ODEFunc()
model2.load_state_dict(torch.load('./msd_nn2.pt'))
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