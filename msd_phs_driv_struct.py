import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import derivnets
import math


epochs = 2000
use_adjoint = True
batch_size = 50
run_time = 25.0
data_size = 250
noise_std = 0.000

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
        self.G = torch.Tensor([[0], [1/self.m]])

    def forward(self, t, x, u):
        dx = self.A.mm(x)  + self.G * u
        return dx

class InputGenerator(nn.Module):
    def __init__(self, w=0.25*math.pi, a=1.5):
        super(InputGenerator, self).__init__()
        self.w = w
        self.a = a

    def forward(self, t):
        return self.a*torch.sin(self.w * t)*torch.exp(-t/3.5)

class ODE_Shell(nn.Module):
    def __init__(self):
        super(ODE_Shell, self).__init__()
        self.plant = MassSpringDamper()
        self.inputFunc = InputGenerator()

    def forward(self,t, x):
        u = self.inputFunc(t)
        dx = self.plant(t, x, u)
        return dx



with torch.no_grad():
    true_x0 = torch.Tensor([[0.0], [0.0]])
    t = torch.linspace(0.0, run_time, data_size)
    true_x = odeint(ODE_Shell(), true_x0, t, method='dopri5')

# add some noise
meas_x = true_x + noise_std * torch.randn(data_size, 2, 1)
meas_x[1,:,:] = true_x[1,:,:]   # no noise on initial state


def get_batch(t, true_x):
    if data_size - batch_size > 0:
        s = np.random.choice(np.arange(data_size-batch_size),replace=False)
    else:
        s = 0
    batch_t = t[s:s+batch_size-1]-t[s]
    batch_x0 = true_x[s, :, 0].unsqueeze(1)
    batch_x = true_x[s:s+batch_size-1, :, 0]
    return batch_x0, batch_t, batch_x

class PHS_Func(nn.Module):
    def __init__(self):
        super(PHS_Func, self).__init__()
        # self.Hnet = derivnets.DerivNet(nn.Linear(2,50), nn.Tanh(), nn.Linear(50,1))
        self.mnet = derivnets.DerivNet(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 25), nn.Tanh(), nn.Linear(25, 1))
        self.vnet = derivnets.DerivNet(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 25), nn.Tanh(), nn.Linear(25, 1))
        # self.dnet = nn.Sequential(nn.Linear(1,50), nn.Tanh(), nn.Linear(50,1))

        self.dnet = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 25), nn.Tanh(), nn.Linear(25, 1))
        self.inputFunc = InputGenerator()

    def forward(self, t, x):
        # H, dHdx = self.Hnet(x.t())
        V, dVdx = self.vnet(x[0].unsqueeze(1))
        sM, dsMdx = self.mnet(x[0].unsqueeze(1))
        sd = self.dnet(x[1])
        # d = self.dnet(x[1].abs())
        dx = torch.empty(2, 1)
        dx[0] = sM[0].pow(2) * x[1]  # q dot
        dx[1] = -x[1] * dsMdx[0] * sM * x[1] - dVdx[0] - sd.pow(2) * sM[0].pow(2) * x[1] + self.inputFunc(t)
        # dx[1] = -dHdx[0] - sd * sd * x[1]
        # dx[1] = -dHdx[0] - x[1] * d
        return dx


model = PHS_Func()
model.load_state_dict(torch.load('./msd_phs_driv.pt'))
model.train()

criterion = torch.nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
train_loss = np.empty([epochs, 1])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 4000], gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50,
#                                                        min_lr=1e-4,
#                                                        factor=0.1,
#                                                        cooldown=25)

for epoch in range(epochs):

    batch_x0, batch_t, batch_x = get_batch(t, meas_x)
    tol = 1e-6
    count = 0
    done = False
    # while True:
        # try:
    optimizer.zero_grad()
    pred_x = odeint(model, batch_x0, batch_t, rtol=tol)
    pred_diff = pred_x[1:-1, 0, 0] - pred_x[0:-2, 0, 0]
    true_diff = batch_x[1:-1, 0] - batch_x[0:-2, 0]
    loss = criterion(true_diff, pred_diff)
    loss.backward()
    optimizer.step()
    # done = True
        # except:
        #     count += 1
        #     if count > 4:
        #         print('could not solve')
        #         break
        #     tol = tol*10
        # if done:
        #     break


    # loss = criterion(batch_x.view(batch_size*2),pred_x.view(batch_size*2))  # somehow have a momentum sensor
    # train only against position state

    train_loss[epoch] = loss.detach().numpy()
    scheduler.step(epoch)
    # scheduler.step(loss)
    print('Epoch ', epoch, ': loss ', loss.item())



# To save trained model
torch.save(model.state_dict(), './msd_phs_driv.pt')

#to load model
# model2 = PHS_Func()
# model2.load_state_dict(torch.load('./msd_phs.pt'))
# model2.eval()

with torch.no_grad():
    pred_x = odeint(model, true_x0, t)






with torch.no_grad():
    fplot, ax = plt.subplots(2, 1, figsize=(4, 6))
    ax[0].plot(np.log(train_loss))

    ax[1].plot(t.numpy(),true_x[:, 0, 0].numpy())
    ax[1].plot(t.numpy(), true_x[:, 1, 0].numpy())
    ax[1].plot(t.numpy(),pred_x[:, 0, 0].detach().numpy(),'--')
    ax[1].plot(t.numpy(), pred_x[:, 1, 0].detach().numpy(),'--')
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('states (x)')
    ax[1].legend(['position','momentum','pred pos', 'pred mom'])
    plt.show()