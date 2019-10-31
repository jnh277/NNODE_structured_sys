import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


epochs = 900
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




def load_exp(i=0):
    data = pd.read_csv('data_set_'+str(i)+'.csv')
    x1 = data['y1']
    n = np.size(x1)
    true_x = torch.empty(n,2,1)
    true_x[:, 0, 0] = torch.from_numpy(x1.values)
    true_x[:, 1, 0] = torch.from_numpy(data['y2'].values)
    true_x0 = true_x[1, :, :]
    t = torch.from_numpy(data['time'].values)
    return t, true_x, true_x0

class DataLoader:
    def __init__(self, exp=0):
        self.t, self.x, self.x0 = load_exp(exp)

    def get_batch(self):
        s = np.random.choice(np.arange(data_size - batch_size), replace=False)
        batch_t = self.t[s:s + batch_size] - self.t[s]
        batch_x0 = self.x[s, :, 0].unsqueeze(1)
        batch_x = self.x[s:s + batch_size, :, 0]
        return batch_x0, batch_t, batch_x


# this will be the validation set
t, true_x, true_x0 = load_exp(0)

# these will be the training sets
datloaders = [DataLoader(1), DataLoader(2), DataLoader(3), DataLoader(4)]

# def get_batch(t, true_x):
#     s = np.random.choice(np.arange(data_size-batch_size),replace=False)
#     batch_t = t[s:s+batch_size]-t[s]
#     batch_x0 = true_x[s, :, 0].unsqueeze(1)
#     batch_x = true_x[s:s+batch_size, :, 0]
#     return batch_x0, batch_t, batch_x

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        dx = torch.empty(2,1)
        dx[0] = x[1]
        dx[1] = self.net(x.t())
        return dx


model = ODEFunc()
criterion = torch.nn.MSELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
train_loss = np.empty([epochs, 1])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50,
#                                                        min_lr=1e-4,
#                                                        factor=0.1,
#                                                        cooldown=25)

for epoch in range(epochs):
    optimizer.zero_grad()
    # choose experiment
    exp = epoch % 4
    # get batch from that experiment
    batch_x0, batch_t, batch_x = datloaders[exp].get_batch()
    pred_x = odeint(model, batch_x0, batch_t)
    loss = criterion(batch_x.view(batch_size*2),pred_x.view(batch_size*2))
    # pred_x = odeint(model, true_x0, t)
    # loss = criterion(true_x.view(data_size*2),pred_x.view(data_size*2))
    loss.backward()
    optimizer.step()
    train_loss[epoch] = loss.detach().numpy()
    scheduler.step()
    # scheduler.step(loss)
    print('Epoch ', epoch, ': loss ', loss.item())



with torch.no_grad():
    pred_x = odeint(model, true_x0, t)


# To save trained model
torch.save(model.state_dict(), './msd_nn_int_known.pt')

with torch.no_grad():
    fplot, ax = plt.subplots(2, 1, figsize=(4, 6))
    ax[0].plot(np.log(train_loss))

    ax[1].plot(t.numpy(),true_x[:, 0, 0].numpy())
    ax[1].plot(t.numpy(), true_x[:, 1, 0].numpy())
    ax[1].plot(t.numpy(),pred_x[:, 0, 0].detach().numpy())
    ax[1].plot(t.numpy(), pred_x[:, 1, 0].detach().numpy())
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('states (x)')
    ax[1].legend(['position','velocity'])
    plt.show()