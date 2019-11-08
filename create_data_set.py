import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np

epochs = 300
use_adjoint = True
batch_size = 100
run_time = 25.0
data_size = 250
noise_std = 0.01

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

for i in range(5):
    # true_x0 = torch.Tensor([[1.0],[0.0]])
    true_x0 = torch.Tensor(2,1)
    true_x0[0] = 2.0*torch.rand(1,1)
    true_x0[1] = 1.0*torch.rand(1,1)
    t = torch.linspace(0.0, run_time, data_size)
    with torch.no_grad():
        true_x = odeint(MassSpringDamper(), true_x0, t, method='dopri5')
    y = true_x.squeeze() + noise_std * torch.randn(data_size, 2)
    data_dict = {"time": t.numpy(),
                 "y1": y[:, 0].numpy(),
                 "y2": y[:, 1].numpy()}

    data = pd.DataFrame(data_dict)
    data.to_csv("data_set_"+str(i)+".csv")




with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 6))

    ax.plot(t.numpy(), true_x[:, 0, 0].numpy())
    ax.plot(t.numpy(), true_x[:, 1, 0].numpy())
    ax.plot(t.numpy(), y[:, 0].numpy(),'*')
    ax.plot(t.numpy(), y[:, 1].numpy(), '*')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('states (x)')
    ax.legend(['position','velocity'])
    plt.show()




