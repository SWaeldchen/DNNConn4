'''
Created on May 26, 2021

@author: stephanw
'''

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle

from src.utils import *
from numpy import inner
from qDataSet import QDataset


class Net(nn.Sequential):
    def __init__(self, innerNet, lr=0):
        super(Net, self).__init__(innerNet)
        self.innerNet = innerNet
        self.lr = lr
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        return self.innerNet.forward(x)
    
class NNLearner():
    def __init__(self, net, channels = 3, lr=0.1, gamma = 0.95, batch_size = 64, trainSet = None, testSet = None):
        self.lr = lr
        self.net = Net(net, self.lr)
        self.channels = channels
        
        self.trainSet = trainSet
        self.testSet = testSet
        
        self.gamma = gamma
        self.batchSize = batch_size
    
    def forward(self, state):
        state = state.type(T.float32)
        return self.net.forward(state)
    
    def eval_on_batch(self, dataSet, batchSize):
        
        # Get Batch
        state_batch, action_batch, nx_state_batch, reward_batch, terminal_batch = dataSet.get_random_batch(batchSize, self.net.device)
        
        #Calculate Q Values
        batch_index = np.arange(self.batchSize, dtype=np.int32)
        
        qValues = self.forward(state_batch)[batch_index, action_batch]
        
        #Calculate Target Values from 
        nx_state_batch = invert_state(nx_state_batch, self.channels) # change perspective
        qValuesOpp = self.forward(nx_state_batch)
        qValuesOpp[terminal_batch] = 0.0
        qTarget = reward_batch - self.gamma*T.max(qValuesOpp,dim=1)[0]
        
        return self.net.loss(qTarget, qValues).to(self.net.device)
        
    
    def learn(self):
        # Set gradients to zero
        self.net.optimizer.zero_grad()
        
        loss = self.eval_on_batch(self.trainSet, self.batchSize)
        
        #Update
        loss.backward()
        self.net.optimizer.step()
        
        return loss.item()
    
    def test(self):
        loss = self.eval_on_batch(self.testSet, self.batchSize)
        return loss.item()
