'''
Created on May 22, 2021

@author: stephanw
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from networks import NNLearner
from qDataSet import QDataset

FCNN1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(7*6*3, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 7),
            )

if __name__ == '__main__':
    
    # set up
    n_steps =  500000
    trainLossList = []
    testLossList = []
    
    trainSet = QDataset().loader('trainset1.pkl')
    testSet = QDataset().loader('testset1.pkl')
    
    # set up DQN_agent and his Opponent
    nnLearner = NNLearner(FCNN1, trainSet=trainSet, testSet=testSet)
    
    for step in range(n_steps):
      
        trainLoss = nnLearner.learn()
        testLoss = nnLearner.test()
        
        if trainLoss:
            trainLossList.append(trainLoss)
        if testLoss:
            testLossList.append(testLoss)
                
        avg_trainLoss = np.mean(trainLossList[-200:])
        avg_testLoss = np.mean(testLossList[-200:])

        if step %20 == 0 and trainLoss != None and testLoss != None:
            print('episode ', step, 'avg_Train_loss %.15f' % avg_trainLoss, 'avg_Test_loss %.5f' % avg_testLoss)

        