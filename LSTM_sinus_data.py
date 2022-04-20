import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTMPredictor
import torch
import torch.nn as nn
import torch.optim as optim

N = 100 # number of samples
L = 1000 # length of each sample
T = 20 # width of wave

x = np.empty((N,L) , np.float32) #why tuple input?
#we shift each sample a little bit
x[:] = np.array(range(L)) + np.random.randint(-4*T,4*T,N).reshape(N,1) #why re-shape?
y = np.sin(x/1.0/T).astype(np.float32) #why specify types?

'''
plt.figure(figsize=(10,8))
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("y")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(x.shape[1]), y[0,:], 'r', linewidth=2.0)
plt.show()
'''

#so we have the first 1000 observations per sample and want to predict the next 1000 steps

#creating the training loop
#we only use the y values
# y = 100,1000
train_input = torch.from_numpy(y[3:, :-1]) # 97,999 -- using from sample 3 and all after - for each excluding the last observation
train_target = torch.from_numpy(y[3:, 1:]) # 97,999 -- targets reach 1 value more into the future enabling training
test_input = torch.from_numpy(y[:3, :-1]) # 3,999 -- testing on other samples
test_target = torch.from_numpy(y[:3, 1:]) # 3,999

model = LSTMPredictor()
criterion = nn.MSELoss() #used for point forecasts

optimizer = optim.LBFGS(model.parameters(), lr=0.8)

n_steps = 10
for i in range(n_steps):
    print("Step", i)
    #defining a closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        out1,out2 = model(train_input)
        loss1 = criterion(out1,train_target)
        loss2 = criterion(out2,train_target)
        print("loss", loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)

    #doing the actual predicitons
    with torch.no_grad():

        future = 1000
        pred = model(test_input, future_preds=future)
        loss = criterion(pred[:, :-future], test_target) #we need to make sure the test and target shapes are similar
        print("loss", loss.item())
        y = pred.detach().numpy()

    plt.figure(figsize=(12, 6))
    plt.title(f"Step {i+1}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    n = train_input.shape[1] #999
    def draw(y_i, color):
        plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
        plt.plot(np.arange(n, n+future), y_i[n:], color + ":" , linewidth=2.0)
    draw(y[0], 'r')
    draw(y[1], 'b')
    draw(y[2], 'g')

    plt.savefig("predict%d.pdf"%i)
    plt.close()
