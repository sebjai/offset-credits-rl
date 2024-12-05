#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:39:31 2024

@author: liamwelsh
"""

# -*- coding: utf-8 -*-

# import os, sys
# sys.path.insert(0, os.path.abspath(".."))

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

import nash_dqn_split, offset_env
import dill
import wandb
import io
from PIL import Image
import numpy as np
import sys
import torch


if torch.cuda.is_available():  
    dev = torch.device("cuda:0")  
else:  
    dev = torch.device("cpu")    


#%%
config={
        'random_seed': 2024,
        'learning_rate': 0.005,
        'gamma': 0.9999,
        'beta': 0,
        'alpha': 0,
        'tau':0.05,
        'sched_step_size': 20,
        'n_nodes': 120,
        'n_layers': 3,
    }

# agent setup
agent_config = {
    1 : {
        'gen_capacity': 0.5,
        'gen_cost': 1.25
        },
    2 : {
        'gen_capacity': 1,
        'gen_cost': 2.5
        },
    3  : {
        'gen_capacity': 0.5,
        'gen_cost': 1.25
        }
}

# environment parameters
env_config = {
    'n_agents' : 3,
    'time_steps': 25,
    'periods': np.array([1/12, 2/12]),

    'gen_capacity': torch.tensor([i['gen_capacity'] for i in agent_config.values()]).to(dev),
    'gen_cost' : torch.tensor([i['gen_cost'] for i in agent_config.values()]).to(dev),
    'gen_impact': 0.05,

    'requirement': torch.tensor([5, 4, 3]).to(dev),
    'penalty': 2.5,
    'penalty_type': 'diff',

    'price_start': 2.5,
    'friction': 0.1,
    'sigma': 0.25,
    
    'decay': 0
}


torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])


env = offset_env.offset_env(T=env_config['periods'], S0=env_config['price_start'], sigma=env_config['sigma'], 
                            kappa = env_config['friction'], 
                            eta = env_config['gen_impact'], 
                            xi = env_config['gen_capacity'], c = env_config['gen_cost'],  
                            R = env_config['requirement'], pen = env_config['penalty'], 
                            n_agents = env_config['n_agents'],
                            N = env_config['time_steps'],
                            penalty = env_config['penalty_type'], decay = env_config['decay'],
                            dev = dev)

obj = nash_dqn_split.nash_dqn_split(env,
                        n_agents = env_config['n_agents'],
                        gamma = config['gamma'], alpha=config['alpha'], beta=config['beta'],
                        lr = config['learning_rate'],
                        tau = config['tau'],
                        sched_step_size=config['sched_step_size'],
                        name = "test", n_nodes = config['n_nodes'], n_layers = config['n_layers'],
                        dev = dev)



obj.train(n_iter=15000, 
          batch_size=1024, 
          n_plot=2500,
          update_type = 'random')
