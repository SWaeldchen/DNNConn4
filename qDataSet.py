'''
Created on May 26, 2021

@author: stephanw
'''
import torch as T
import numpy as np
import os
import pickle

from src.utils import *

class QDataset():
    def __init__(self, size=1000, channels=1):
        self.size = size
        self.cntr = 0
        self.isfull = False
        self.channels = channels
        
        self.states = np.zeros((self.size, 6,7, channels), dtype=np.int8)
        self.nx_states = np.zeros((self.size, 6,7, channels), dtype=np.int8)
        self.actions = np.zeros(self.size, dtype=np.int8)
        self.rewards = np.zeros(self.size, dtype=np.int8)
        self.terminals = np.zeros(self.size, dtype=np.bool)
    
    def add(self, state, action, nx_state, reward, terminal):
        index = self.cntr % self.size
        self.states[index] = state
        self.nx_states[index] = nx_state
        self.rewards[index] = reward
        self.actions[index] = action
        self.terminals[index] = terminal
        self.cntr += 1
        
        if self.isfull==False and self.cntr==self.size:
            self.isfull = True
    
    def get_random_batch(self, batchSize, device):   
        if self.cntr < batchSize:
            return
        
        # Random Indices
        max_set = min(self.size, self.cntr)
        batch = np.random.choice(max_set, batchSize, replace=False)

        # Load Q-Fkt inputs from trainSet
        state_batch = T.tensor(self.states[batch]).to(device)
        nx_state_batch = T.tensor(self.nx_states[batch]).to(device)
        action_batch = self.actions[batch]
        reward_batch = T.tensor(self.rewards[batch]).to(device)
        terminal_batch = T.tensor(self.terminals[batch]).to(device)
        
        return state_batch, action_batch, nx_state_batch, reward_batch, terminal_batch
        
        
    def save(self, file_name = "qDataset.pkl"):
        directory = os.getcwd()
        file_path = os.path.join(directory,"saved_data_sets", file_name)
        pickle.dump(self, open(file_path, 'wb'))
        
    @classmethod
    def loader(cls,file_name = "qDataset.pkl"):
        directory = os.getcwd()
        file_path = os.path.join(directory,"saved_data_sets", file_name)
        return pickle.load(open(file_path, 'rb'))