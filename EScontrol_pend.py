import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import derivnets


epochs = 500
use_adjoint = True
batch_size = 30
run_time = 5.0
data_size = 500
noise_std = 0.01

if use_adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class pendSys(nn.Module):
    def __init__(self, m=0.5, l=0.3, J=0.1, b=0.2, g=9.8):
        super(pendSys, self).__init__()
        self.m = m
        self.l = l
        self.J = J
        self.b = b
        self.g = g
        self.F = torch.Tensor([[0, 1], [-1, -b]])
        self.G = torch.Tensor([[0], [1]])
        self.Hnet = derivnets.DerivNet(nn.Linear(2,50), nn.Tanh(), nn.Linear(50,1))

    def forward(self, t, x):
        dHdx = torch.empty(2, 1)
        dHdx[0] = -self.m*self.g*self.l*torch.sin(x[0])
        dHdx[1] = x[1]/(self.J + self.m*self.l*self.l)

        Ha, dHadx = self.Hnet(x.t())
        # print(dHadx)

        input = torch.empty(2, 1)
        input[0] = 0
        input[1] = -dHadx[0]

        # dx = self.F.mm(dHdx)
        # print(dx.size())
        dx = self.F.mm(dHdx) + input
        # dx = self.F.mm(dHdx)
        return dx

model = pendSys()
batch_t = torch.linspace(0.0, run_time, data_size)

with torch.no_grad():
    # batch_x0 = np.random.rand(2, 1)
    true_x0 = torch.Tensor([[np.random.rand()], [np.random.rand()]])
    t = torch.linspace(0.0, run_time, data_size)
    true_x = odeint(model, true_x0, batch_t, method='dopri5')

plt.plot(t,true_x[:, :, 0])

criterion = torch.nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
train_loss = np.empty([epochs, 1])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50,
#                                                        min_lr=1e-4,
#                                                        factor=0.1,
#                                                        cooldown=25)


batch_x = torch.zeros(true_x[:, 0, 0].size())
print(batch_x.size())
for epoch in range(epochs):
    optimizer.zero_grad()
    # batch_x0, batch_t, batch_x = get_batch(t, meas_x)
    batch_x0 = torch.Tensor([[np.random.rand()], [np.random.rand()]])
    pred_x = odeint(model, batch_x0, batch_t)
    x_new = pred_x.clone().detach()
    nt = x_new.size(0)
    H = []
    dHdx = []
    for i in range(nt):
        Hi, dHdxi = model.Hnet(x_new[i, :, :].t())
        dHdx.append(dHdxi)
    # loss = criterion(batch_x.view(batch_size*2),pred_x.view(batch_size*2))  # somehow have a momentum sensor
    # train only against position state
    # print(pred_x[:, 0, 0])
    # dev = Gp*F*dHadx
    loss = criterion(batch_x, pred_x[:, 0, 0])
    loss.backward()
    optimizer.step()
    train_loss[epoch] = loss.detach().numpy()
    scheduler.step()
    # scheduler.step(loss)
    print('Epoch ', epoch, ': loss ', loss.item())


# To save trained model
torch.save(model.state_dict(), './pend_nn1.pt')


true_x0 = torch.Tensor([[np.random.rand()], [np.random.rand()]])
with torch.no_grad():
    pred_x = odeint(model, true_x0, t)

with torch.no_grad():
    fplot, ax = plt.subplots(2, 1, figsize=(4, 6))
    ax[0].plot(np.log(train_loss))
#    ax[1].plot(t.numpy(),true_x[:, 0, 0].numpy())
#    ax[1].plot(t.numpy(), true_x[:, 1, 0].numpy())
    ax[1].plot(t.numpy(),pred_x[:, 0, 0].detach().numpy())
    ax[1].plot(t.numpy(), pred_x[:, 1, 0].detach().numpy())
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('states (x)')
    ax[1].legend(['position','velocity'])
    plt.show()