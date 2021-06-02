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
from networks import Net

class Player():
    def __init__(self):
        self.action_space = [i for i in range(7)]
        self.position = 0
    def choose_action(self, observation, action_mask):
        pass

class RandomPlayer(Player):
    def __init__(self):
        super().__init__()
    def choose_action(self, observation, action_mask):
        """ Provides a random_move that is guaranteed to be legal, given a valid action_mask """
        if action_mask[-1] == 1:
            return 7
        else:
            action_mask = np.copy(action_mask[:-1])
            assert np.shape(np.array(self.action_space)) == np.shape(action_mask)
            possible_actions = np.array(self.action_space)[action_mask!=0]
            action = np.random.choice(possible_actions)

            return action

class MinMaxPlayer(Player):
    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth
    def choose_action(self, observation, action_mask):
        if action_mask[-1] == 1:
            return 7
        else:
            possible_actions = list(np.array(self.action_space)[action_mask[:-1]!=0])
            action = minimax_action(np.flipud(observation[self.position^1]["board"]), possible_actions, self.depth)
        return action

class NNPlayer(Player):
    def __init__(self, net, epsilon = 0.1, explore = True, channels = 3):
        super().__init__()
        
        self.net = Net(net)
        self.epsilon = epsilon
        self.explore = explore
        self.channels = channels
    
    def choose_action(self, observation, action_mask):
        if action_mask[-1] == 1:
            return 7
        else:
            action_mask = np.copy(action_mask[:-1])
        
        if np.random.random() > self.epsilon or self.explore == False:
            state = board_to_state(observation[self.position]['board'], channels = self.channels)
            actionValues = self.calc_values(state) # Gets Numpy array from torch tensor
            if np.min(actionValues)< 0.0:
                actionValues -= (np.min(actionValues))*2 # Shift the smallest value above 0, so the minimal action is always valid.
            action = np.argmax(actionValues * action_mask)
        else:
            assert np.shape(np.array(self.action_space)) == np.shape(action_mask)
            possible_actions = np.array(self.action_space)[action_mask!=0]
            action = np.random.choice(possible_actions)

        return action
    
    def calc_values(self, state):
        state = T.tensor([state]).to(self.net.device)
        state = state.type(T.float32)
        return self.net.forward(state).detach().numpy()[0]