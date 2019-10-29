import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import numpy as np

use_adjoint = True

batch_size = 100
run_time = 25.0
data_size = 250

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
model2.load_state_dict(torch.load('./msd_nn.pt'))
model2.eval()

def get_batch():
    s = np.random.choice(np.arange(data_size-batch_size),replace=False)
    batch_t = t[s:s+batch_size]-t[s]
    batch_x0 = true_x[s, :, 0].unsqueeze(1)
    batch_x = true_x[s:s+batch_size, :, 0].unsqueeze(2)
    return batch_x0, batch_t, batch_x


batch_x0, batch_t, batch_x = get_batch()


with torch.no_grad():
    pred_x = odeint(model2, batch_x0, batch_t, method='dopri5')


with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 6))
    ax.plot(batch_t.numpy(),batch_x[:, 0, 0].numpy())
    ax.plot(batch_t.numpy(), batch_x[:, 1, 0].numpy())
    ax.plot(batch_t.numpy(),pred_x[:, 0, 0].detach().numpy(),'-.')
    ax.plot(batch_t.numpy(), pred_x[:, 1, 0].detach().numpy(),'-.')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('states (x)')
    ax.legend(['pos true','vel true','pos model', 'vel model'])
    plt.show()