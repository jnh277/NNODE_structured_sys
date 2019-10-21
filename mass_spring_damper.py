import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import numpy as np

epochs = 100
use_adjoint = True
batch_size = 100
run_time = 25.0
data_size = 250

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

true_x0 = torch.Tensor([[1.0],[0.0]])
t = torch.linspace(0.0, run_time, data_size)


with torch.no_grad():
    true_x = odeint(MassSpringDamper(), true_x0, t, method='dopri5')


def get_batch():
    s = np.random.choice(np.arange(data_size-batch_size),replace=False)
    batch_t = t[s:s+batch_size]
    batch_x0 = true_x[s, :, 0].unsqueeze(1)
    batch_x = true_x[s:s+batch_size, :, 0]
    return batch_x0, batch_t, batch_x

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


model = ODEFunc()
criterion = torch.nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loss = np.empty([epochs, 1])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50,
                                                       min_lr=1e-10,
                                                       factor=0.5,
                                                       cooldown=15)

for epoch in range(epochs):
    optimizer.zero_grad()
    # batch_x0, batch_t, batch_x = get_batch()
    # pred_x = odeint(model, batch_x0, batch_t)
    # loss = criterion(batch_x.view(batch_size*2),pred_x.view(batch_size*2))
    pred_x = odeint(model, true_x0, t)
    loss = criterion(true_x.view(data_size*2),pred_x.view(data_size*2))
    loss.backward()
    optimizer.step()
    train_loss[epoch] = loss.detach().numpy()
    scheduler.step(loss)
    print('Epoch ', epoch, ': loss ', loss.item())

with torch.no_grad():
    pred_x = odeint(model, true_x0, t)

with torch.no_grad():
    fplot, ax = plt.subplots(2, 1, figsize=(4, 6))
    ax[0].plot(train_loss)

    ax[1].plot(t.numpy(),true_x[:, 0, 0].numpy())
    ax[1].plot(t.numpy(), true_x[:, 1, 0].numpy())
    ax[1].plot(t.numpy(),pred_x[:, 0, 0].detach().numpy())
    ax[1].plot(t.numpy(), pred_x[:, 1, 0].detach().numpy())
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('states (x)')
    ax[1].legend(['position','velocity'])
    plt.show()