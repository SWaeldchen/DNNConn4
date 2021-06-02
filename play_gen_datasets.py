'''
Created on May 19, 2021

@author: stephanw
'''

import gym_connect4

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from players import NNPlayer, RandomPlayer, MinMaxPlayer
from qDataSet import QDataset

import random
from src.utils import *

FCNN1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3*7*6, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 7),
            )

if __name__ == '__main__':
    
    
    # set up environment
    NUMBER_OF_GAMES =  50000
    env, scores, eps_history, n_games = setup_game(NUMBER_OF_GAMES)
    losses = []
    observation = env.reset()
    
    # Choose Hero and Opponent
    hero = NNPlayer(FCNN1)
    #hero = RandomPlayer()
    #hero = MinMaxPlayer(depth = 1)
    #opponent = RandomPlayer()
    opponent = MinMaxPlayer(depth = 1)
    players = [hero, opponent]
    print(players)
    
    # Fill Datasets
    size = 2000
    channels = 3
    trainData = QDataset(size, channels)
    testData = QDataset(size, channels)
    
    
    for i in range(n_games):
        
        # Select player order
        random.shuffle(players)
        players[0].position = 0 # inform players of their order
        players[1].position = 1
        
        # Set up new game
        observation = env.reset() # environment is reset
        game_over = False
        score = 0
        active_player = 0
        
        while not game_over:
            
            action_mask_p0 = observation[0]["action_mask"] #Gives possible Actions for Player 1
            action_p0 = players[0].choose_action(observation, action_mask=action_mask_p0)
            
            action_mask_p1 = observation[1]["action_mask"] #Gives possible Actions for Player 2
            action_p1 = players[1].choose_action(observation, action_mask=action_mask_p1)
            
            action_dict = [action_p0, action_p1]
            nx_observation, rewards, game_over, info = env.step(action_dict)
            score += rewards[hero.position] # add rewards to score for agent 0
            
            #print(observation[0]["board"])
            
            # Fill data sets
            if (game_over == True) or np.random.rand()<0.1:
                state = board_to_state(observation[active_player]["board"], channels)
                nx_state = board_to_state(nx_observation[active_player]["board"], channels)
                
                if np.random.rand()<0.5:
                    trainData.add(state,action_dict[active_player],nx_state,rewards[active_player],game_over)
                    if trainData.cntr % 50 == 0:
                        print(trainData.cntr)
                else:
                    testData.add(state,action_dict[active_player],nx_state,rewards[active_player],game_over)
                    if testData.cntr % 50 == 0:
                        print(testData.cntr)
                    
             
            observation = nx_observation   
            active_player ^= 1
            
            # if (testData.cntr % 100 == 0):
            #     print(testData.cntr)
        
        if (trainData.isfull and testData.isfull):
            break
    
    trainData.save('trainset1.pkl')
    testData.save('testset1.pkl')
    