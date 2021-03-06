import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import derivnets


epochs = 10
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

        input = torch.empty(2, 1)
        input[0] = 0
        input[1] = -dHadx[1]

        dx = self.F.mm(dHdx) + input
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

# Define target equilibrium and build vector to compare to ODE sover ouput
u_star = 5
tar_eq = torch.Tensor([[model.C*u_star, 0], [0, (model.L/model.R)*u_star]])
reference_x = torch.ones(true_x[:, :, 0].size())
reference_x = reference_x.mm(tar_eq)

# preallocate tensors for constraint component of the loss function
constraint_tensor2 = torch.zeros(true_x[:, 0, 0].size())
reference_zero = torch.zeros(true_x[:, 0, 0].size())

for epoch in range(epochs):
    optimizer.zero_grad()
    # batch_x0, batch_t, batch_x = get_batch(t, meas_x)
    batch_x0 = torch.Tensor([[np.random.rand()], [np.random.rand()]])
    pred_x = odeint(model, batch_x0, batch_t)

    x_new = pred_x.clone().detach()
    # x_new = pred_x
    nt = x_new.size(0)
    # H = []
    # dHdx = []
    # dHdx = torch.ones(true_x[:, 0, 0].size())
    Hi, dHdxi = model.Hnet(x_new[:, :, 0])
    constraint_tensor = dHdxi[1] - dHdxi[0]/model.R

    # Construct loss function---contains penalty for deviation from equilibrium and violating G perp PDE
    # loss = criterion(reference_x[:,0], pred_x[:, 0, 0]) + criterion(reference_x[:,1], pred_x[:, 1, 0]) # + criterion(constaint_tensor, reference_zero)
    loss = criterion(reference_x, pred_x[:,:,0]) + criterion(constraint_tensor.squeeze(), reference_zero) #
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
# To save trained model
torch.save(model.state_dict(), './nn_ctrl_elec.pt')
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