import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import derivnets


# ode solver options
use_adjoint = True

# specify how many sets of data to use (i.e. number of experiments/trials run)
num_data_sets = 6

# som training options, setting batch_size_min = 500 will force it to use entire set each time
epochs = 500
training_milestones = [100,200,300,400,500]        # epochs at which to reduce learning rate
batch_size_max = 400
batch_size_min = 100
lr_start = 0.01
gamma = 0.5     # learning rate reduction factor

# add in absolute difference to cost along with a weighting factor
abs_weighting = 1e-3

# Some simulation options
run_time = 5.0
data_size = 500

# currently no noise on data (option to add noise not implemented)
noise_std = 0.00

if use_adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# define pendulum dynamics
# to make conservative we set b=0 (default)
class pendSys(nn.Module):
    def __init__(self, m=0.5, l=0.3, J=0.1, b=0.0, g=9.8):
        super(pendSys, self).__init__()
        self.m = m
        self.l = l
        self.J = J
        self.b = b
        self.g = g
        self.F = torch.Tensor([[0, 1], [-1, -b]])
        self.G = torch.Tensor([[0], [1]])
        # self.Hnet = derivnets.DerivNet(nn.Linear(2,50), nn.Tanh(), nn.Linear(50,1))

    def forward(self, t, x):
        dHdx = torch.empty(2, 1)
        dHdx[0] = -self.m*self.g*self.l*torch.sin(x[0])
        dHdx[1] = x[1]/(self.J + self.m*self.l*self.l)

        # Ha, dHadx = self.Hnet(x.t())
        # print(dHadx)

        # input = torch.empty(2, 1)
        # input[0] = 0
        # input[1] = -dHadx[0]

        # dx = self.F.mm(dHdx)
        # print(dx.size())
        dx = self.F.mm(dHdx)
        # dx = self.F.mm(dHdx)
        return dx

with torch.no_grad():
    true_x0 = torch.Tensor([[-1.0], [0.0]])
    t = torch.linspace(0.0, run_time, data_size)
    true_x = odeint(pendSys(), true_x0, t, method='dopri5')

# plot a test simulation
plt.plot(t.numpy(),true_x[:,0].numpy())
plt.plot(t.numpy(),true_x[:,1].numpy())
plt.xlabel('t')
plt.legend(['q','p'])
plt.title('Test simulation of pendulum system')
plt.show()

# simulate some extra data sets
data_sets = []
for i in range(num_data_sets):
    set_x0 = torch.Tensor([[np.random.uniform(low=-1.5, high=1.5)], [0.0]])
    set_x = odeint(pendSys(), true_x0, t, method='dopri5')
    data_sets.append(set_x)

# from the simulated data we want to be able to select out training batches
# the training batches should start at t=0 (to give known initial conditions)
# should have a random length (up to batch size)


def get_batch(t, meas_x):
    batch_size = np.random.choice(np.arange(batch_size_min,batch_size_max),replace=False)
    batch_t = t[0:batch_size]
    batch_x0 = meas_x[0,:,0].unsqueeze(1)
    batch_x = meas_x[0:batch_size, :, 0]
    return batch_x0, batch_t, batch_x

# define a conservative PHS NN to learn this with, using mechanical interconnection structure
# class PHS_Func(nn.Module):
#     def __init__(self):
#         super(PHS_Func, self).__init__()
#         self.Hnet = derivnets.DerivNet(nn.Linear(2,50), nn.Tanh(), nn.Linear(50,1))
#
#     def forward(self, t, x):
#         H, dHdx = self.Hnet(x.t())
#         # print(x[0].unsqueeze(1).size())
#         dx = torch.empty(2, 1)
#         dx[0] = dHdx[1]  # q dot
#         dx[1] = -dHdx[0]
#
#         return dx

class PHS_Func(nn.Module):
    def __init__(self):
        super(PHS_Func, self).__init__()
        self.Vnet = derivnets.DerivNet(nn.Linear(1,50), nn.Tanh(), nn.Linear(50,1))
        self.Mnet = nn.Sequential(nn.Linear(1,10),nn.Linear(10,1), nn.Sigmoid())        # the sigmoid ensures positive output

    def forward(self, t, x):
        V, dVdx = self.Vnet(x[0].unsqueeze(1))
        m = self.Mnet(x[0].unsqueeze(1))
        dx = torch.empty(2, 1)
        dx[0] = x[1]/m  # q dot
        dx[1] = -dVdx[0]

        return dx

model = PHS_Func()


# define criterior, optimiser and scheduler
criterion = torch.nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.5
optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
train_loss = np.empty([epochs, 1])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_milestones, gamma=gamma, last_epoch=-1)


# train model
for epoch in range(epochs):
    # select which trial to use
    set_order = np.random.choice(np.arange(0, num_data_sets),num_data_sets, replace=False)
    train_loss_epoch = 0
    for set_ind in set_order:
        optimizer.zero_grad()
        batch_x0, batch_t, batch_x = get_batch(t, data_sets[set_ind])
        pred_x = odeint(model, batch_x0, batch_t)

        # loss = criterion(batch_x.view(batch_size*2),pred_x.view(batch_size*2))  # somehow have a momentum sensor
        # train only against position state
        pred_diff = pred_x[1:-1, 0, 0] - pred_x[0:-2, 0, 0]
        true_diff = batch_x[1:-1, 0] - batch_x[0:-2, 0]
        loss = criterion(true_diff, pred_diff) + abs_weighting*criterion(pred_x[:,0,0], batch_x[:,0])
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.detach().numpy()/num_data_sets
    train_loss[epoch] = train_loss_epoch
    scheduler.step()
    # scheduler.step(loss)
    print('Epoch ', epoch, ': loss ', train_loss_epoch)


# predict with the model
with torch.no_grad():
    pred_x = odeint(model, true_x0, t)

with torch.no_grad():
    fplot, ax = plt.subplots(2, 1, figsize=(4, 6))
    ax[0].plot(np.log(train_loss))

    plt.subplot(2,1,1)
    plt.plot(np.log(train_loss))
    plt.title('Training loss')

    plt.subplot(2,1,2)

    plt.plot(t.numpy(),true_x[:, 0, 0].numpy())
    plt.plot(t.numpy(), true_x[:, 1, 0].numpy())
    plt.plot(t.numpy(),pred_x[:, 0, 0].detach().numpy(),'--')
    plt.plot(t.numpy(), pred_x[:, 1, 0].detach().numpy(),'--')
    plt.xlabel('time (s)')
    plt.ylabel('states (x)')
    plt.legend(['q true','p true','q pred','p_pred'])
    plt.show()