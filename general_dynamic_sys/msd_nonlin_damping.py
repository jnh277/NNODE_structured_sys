import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import numpy as np

epochs = 300
use_adjoint = True
batch_size = 100
run_time = 15.0
data_size = 150

# damping parameters
c0 = 0.05

if use_adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def damping(v, c0=0.1, c1=-0.2, c2=0.3):
    s = 2.0*v.sign()-1.0      # gives 1 if pos, -1 if negative
    out = v.clone()
    out[s>0] = c0 + c1*v[s>0] + c2*v[s>0].pow(2)
    out[s<=0] = -c0 + c1*v[s<=0] - c2*v[s<=0].pow(2)
    return out



class MassSpringDamper(nn.Module):
    def __init__(self, m=1.0, k=1.0):
        super(MassSpringDamper, self).__init__()
        self.m = m
        self.k = k

    def forward(self, t, x):
        dx = torch.Tensor(2,1)
        dx[0] = x[1]
        dx[1] = (-self.k*x[0] - damping(x[1]))/self.m
        return dx


true_x0 = torch.Tensor([[0.5],[0.0]])
t = torch.linspace(0.0, run_time, data_size)


with torch.no_grad():
    true_x = odeint(MassSpringDamper(), true_x0, t, method='rk4')


class ODEFunc(nn.Module):
    def __init__(self, m=1.0, k=1.0):
        super(ODEFunc, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(1, 25),
        #     nn.Tanh(),
        #     nn.Linear(25, 10),
        #     nn.Tanh(),
        #     nn.Linear(10, 1),
        #     nn.ReLU()
        # )
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.k = k
        self.m = m

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)
    def damping_net(self, v):
        s = v.sign()

        return s*self.net(v.abs()).abs()

    def forward(self, t, x):
        s = x[1].sign()
        dx = torch.Tensor(2, 1)
        dx[0] = x[1]
        # dx[1] = (-self.k*x[0] - s*self.net(x[1].abs()).abs())/self.m
        dx[1] = (-self.k * x[0] - self.damping_net(x[1])) / self.m
        return dx

model = ODEFunc()


criterion = torch.nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loss = np.empty([epochs, 1])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50,
#                                                        min_lr=1e-4,
#                                                        factor=0.1,
#                                                        cooldown=25)

for epoch in range(epochs):
    optimizer.zero_grad()
    # batch_x0, batch_t, batch_x = get_batch()
    # pred_x = odeint(model, batch_x0, batch_t)
    # loss = criterion(batch_x.view(batch_size*2),pred_x.view(batch_size*2))
    pred_x = odeint(model, true_x0, t, method='rk4')
    loss = criterion(true_x.view(data_size*2),pred_x.view(data_size*2))
    loss.backward()
    optimizer.step()
    train_loss[epoch] = loss.detach().numpy()
    scheduler.step(epoch)
    print('Epoch ', epoch, ': loss ', loss.item())

with torch.no_grad():
    pred_x = odeint(model, true_x0, t, method='rk4')

v = torch.linspace(-1.5,1.5,100)
d = damping(v)

with torch.no_grad():
    # s = v.sign()
    # pred_d = s.unsqueeze(1)*model.net(v.unsqueeze(1).abs()).abs()
    pred_d = model.damping_net(v.unsqueeze(1))

with torch.no_grad():
    pred_x = odeint(model, true_x0, t, method='rk4')

with torch.no_grad():
    fplot, ax = plt.subplots(3, 1, figsize=(4, 9))
    ax[0].plot(v.numpy(),d.numpy())
    ax[0].plot(v.numpy(),pred_d.detach().numpy(),'-.')
    ax[0].set_title('Damping function')
    ax[0].set_ylabel('Damping force (N)')
    ax[0].set_xlabel('Velocity (m/s)')
    ax[0].legend(['True', 'model'])

    ax[1].plot(t.numpy(),true_x[:, 0, 0].numpy())
    ax[1].plot(t.numpy(), true_x[:, 1, 0].numpy())
    ax[1].plot(t.numpy(),pred_x[:, 0, 0].detach().numpy())
    ax[1].plot(t.numpy(), pred_x[:, 1, 0].detach().numpy())
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('states (x)')
    ax[1].legend(['position','velocity'])

    ax[2].plot(np.log(train_loss))
    plt.show()