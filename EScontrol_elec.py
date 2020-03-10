import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import derivnets


epochs = 1
use_adjoint = True
batch_size = 30
run_time = 0.5
data_size = 250
noise_std = 0.01

if use_adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class RCLcircuit(nn.Module):
    def __init__(self, R=5.0, C=0.01, L=0.005):
        super(RCLcircuit, self).__init__()
        self.R = R
        self.C = C
        self.L = L
        self.F = torch.Tensor([[-1/self.R, 1], [-1, 0]])
        self.G = torch.Tensor([[0], [1]])
        self.Hnet = derivnets.DerivNet(nn.Linear(2,50), nn.Tanh(), nn.Linear(50,1))

    def forward(self, t, x):
        dHdx = torch.empty(2, 1)
        dHdx[0] = x[0]/self.C
        dHdx[1] = x[1]/self.L

        Ha, dHadx = self.Hnet(x.t())
        # print(dHadx)

        input = torch.empty(2, 1)
        input[0] = 0
        input[1] = dHadx[1]

        # dx = self.F.mm(dHdx)
        # print(dx.size())
        dx = self.F.mm(dHdx) + input
        # dx = self.F.mm(dHdx)
        return dx

model = RCLcircuit()
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
#
#
#
# with torch.no_grad():
#     pred_x = odeint(model, true_x0, t)
#
#
# # To save trained model
# # torch.save(model.state_dict(), './msd_nn2.pt')
#
# with torch.no_grad():
#     fplot, ax = plt.subplots(2, 1, figsize=(4, 6))
#     ax[0].plot(np.log(train_loss))
#
#     ax[1].plot(t.numpy(),true_x[:, 0, 0].numpy())
#     ax[1].plot(t.numpy(), true_x[:, 1, 0].numpy())
#     ax[1].plot(t.numpy(),pred_x[:, 0, 0].detach().numpy())
#     ax[1].plot(t.numpy(), pred_x[:, 1, 0].detach().numpy())
#     ax[1].set_xlabel('time (s)')
#     ax[1].set_ylabel('states (x)')
#     ax[1].legend(['position','velocity'])
#     plt.show()