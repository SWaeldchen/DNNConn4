from src.minmax import minimax_action
import gym

import torch
import numpy as np
import random
from torch.testing._internal.common_distributed import tmp_dir


def setup_game(n_games = 1500):
    env = gym.make('Connect4Env-v0')
    scores, eps_history = [], []
    n_games = n_games
    return env, scores, eps_history, n_games

def board_to_state(board, channels = 1):
    # convert to 1, -1 boards
    state = np.zeros((board.shape[0], board.shape[1], channels))
    
    if (channels == 1):
        state[board==1,0] = 1
        state[board==2,0] = -1
    elif(channels == 2):
        state[board==1,0] = 1
        state[board==2,0] = -1
        state[board==0,1] = 1
    else:
        state[board==1,0] = 1
        state[board==2,1] = 1
        state[board==0,2] = 1
    
    return state

def invert_state(state, channels=1):
    # inverts the numbers of a board --> change perspective
    
    if (channels == 1):
        state = -state
    elif(channels == 1):
        state[:,0] = -state[:,0]
    else:
        tmp = state[:,0]
        state[:,0] = state[:,1]
        state[:,1] = tmp
        
    return state

def three_channel_board(board):
    # gives a three channel representation of the board: position of players tokens, position of opponenent tokens and emtpy positions
    
    state = np.zeros((board.shape[0], board.shape[1], 3))
    
    state[:,:,0] = np.array(board==1, dtype = np.int8)
    state[:,:,1] = np.array(board==2, dtype = np.int8)
    state[:,:,2] = np.array(board==0, dtype = np.int8)
    
    return state

def invert_three_channel_board(board):
    # invert three dimensional boards for conv_agent
    invert_board = np.zeros((board.shape))

    invert_board[:,0] = board[:,1]
    invert_board[:,1] = board[:,0]
    invert_board[:,2] = board[:,2]
    

    return invert_board


def determine_starting_player(agents_list, DQN_agent_name, Opponent_name):
    # determine starting player and piece of DQN_agent and Opponent
    random.shuffle(agents_list)
    DQN_index = agents_list.index(DQN_agent_name)
    DQN_piece = DQN_index +1
    Opponent_index = agents_list.index(Opponent_name)
    Opponent_piece = agents_list.index(Opponent_name) +1

    return agents_list, DQN_index, DQN_piece, Opponent_index, Opponent_piece


opponent_list = [1,2] # list of possible opponent: 1 == MINIMAX, 2 == RANDOM_AGENT

def select_opponent(opponent_list):
    while True:
        try:
            x = int(input("Please choose an opponent: 1 for MINIMAX, 2 for RANDOM_AGENT"))
            if x in opponent_list:
                return x
            else:
                raise ValueError("Please enter one of the possible numbers for your opponent of choice : {}".format(*[opponent_list]))
        except:
            print("Please enter one of the possible numbers for your opponent of choice : {}".format(*[opponent_list]))

def opponent_action(obs, opp_choice, opp_list, agent, opp_index, action_mask):
    if opp_choice ==1:
        possible_actions = list(np.array(agent.action_space)[action_mask[:-1]!=0])
        action = minimax_action(np.flipud(obs[opp_index^1]["board"]), possible_actions)

    elif opp_choice == 2:
        action = agent.random_action(action_mask)
    else:
        raise ValueError(f"{opp_choice} not in {opp_list}")

    return action



if __name__ =="__main__":
    pass