import warnings
import numpy as np
from generate_synth_data import *
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

'''
simple RNN-based NN model with two LSTM cells outputting the mean and shape parameters of NB distributions
through an log-sigmoid decoder

'''

class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=64):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        #self.linear = nn.Linear(self.n_hidden, 1)
        self.logsigmoid = nn.LogSigmoid(self.n_hidden) #projection from the hidden state to the domain of E[Q_i]

    def forward(self, y, future_preds=0): #future_preds=0 only training on known values - no predictions
        outputs, n_samples = [], y.size(0)
        # initial hidden state - cell 1
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.int)
        # initial cell state - cell 1
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.int)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.int)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.int)

        for time_step in y.split(1, dim=1):
            #N,1
            #going through the y-tensor 1by1 - note the input_size = 1 in LSTMCell1
            #re-estimating the h_t , c_t parameters
            h_t,c_t = self.lstm1(time_step, (h_t,c_t))
            #using the output states from cell 1 to cell 2 - while using its unknown states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            #apply linear layer to get prediction - inputted the output from cell 2
            #output = self.linear(h_t2)  # output from the last FC layer
            output1 = self.logsigmoid(h_t2) #mean parameter
            output2 = self.logsigmoid(h_t2) #shape parameter
            outputs.append([output1,output2])

        #if future > 0 - ie we want to make predictions
        for i in range(future_preds):
            #using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            #output = self.linear(h_t2)
            output1 = self.logsigmoid(h_t2)
            output2 = self.logsigmoid(h_t2)
            outputs.append([output1, output2])

        # transform list to tensor
        output1 = torch.cat(output1, dim=1)
        output2 = torch.cat(output2, dim=1)
        return output1,output2