import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTMPredictor
import torch
import torch.nn as nn
import torch.optim as optim
from generate_synth_data import generate_synth_series

N = 100 # number of samples
L = 50 # length of each sample ie # of issue points/interarrivals

#load generated data
Q_list = []
for seed in range(N):
    y_i, Q_list_i, M_list_i, p_series_Q_i, p_series_M_i = generate_synth_series(N,L,seed)
    Q_list.append(Q_list_i)

#setup traning/test sets
# Q_list = 100,50
Q_array = np.array(Q_list)

#kan modellen trænes ved kun at se på den næste værdi i fremtiden når systemet er så kaotisk?
'''
train_input = torch.IntTensor(Q_list[:-34][:-3]) # 97,49 -- using from sample 3 and all after - for each excluding the last observation
train_target = torch.IntTensor(Q_list[3:][1:]) # 97,49 -- targets reach 1 value more into the future enabling training
test_input = torch.IntTensor(Q_list[3:][:-1]) # 3,49 -- testing on other samples
test_target = torch.IntTensor(Q_list[:3][1:]) # 3,49
'''
train_input = torch.from_numpy(Q_array[3:, :-1]) # 97,49
train_target = torch.from_numpy(Q_array[3:, 1:]) # 97,49
test_input = torch.from_numpy(Q_array[:3, :-1]) # 3,49
test_target = torch.from_numpy(Q_array[:3, 1:]) # 3,49

#setup training loop
model = LSTMPredictor()
#criterion = nn.NLLLoss() #using negative log likelihood - doesnt work for NB distribution
criterion = #some approbiate negative log likelihood fuction based on NB distribution

optimizer = optim.LBFGS(model.parameters(), lr=0.8)

n_steps = 10
for i in range(n_steps):
    print("Step", i)
    #defining a closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        out1,out2 = model(train_input)
        loss = criterion(out,train_target)
        print("loss", loss.item())
        loss.backward()
        print(loss)
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
