# -*- coding: utf-8 -*-

# import os, sys
# sys.path.insert(0, os.path.abspath(".."))

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

import nash_dqn, offset_env
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


#%% Four Agent
config={
        'random_seed': 2024,
        'learning_rate': 0.004,
        'gamma':1,
        'beta': 0,
        'alpha': 0.0,
        'tau':0.05,
        'sched_step_size': 25,
        'n_nodes': 120,
        'n_layers': 5,
    }

# agent setup
agent_config = {
    1 : {
        'gen_capacity': 2,
        'gen_cost': 100,
        'req': 25,
        'count': 1
        },
    2 : {
        'gen_capacity': 1.5,
        'gen_cost': 75,
        'req': 25,
        'count': 1
        },
    3 : {
        'gen_capacity': 1,
        'gen_cost': 50,
        'req': 25,
        'count': 1
        },
    
    4 : {
        'gen_capacity': 0.5,
        'gen_cost': 25,
        'req': 25,
        'count': 1
        }
    
}

gen_cost =  torch.tensor([i['gen_cost'] for i in agent_config.values()]).to(dev)
gen_cap =  torch.tensor([i['gen_capacity'] for i in agent_config.values()]).to(dev)
req_amt =  torch.tensor([i['req'] for i in agent_config.values()]).to(dev)

ag_count = torch.tensor([i['count'] for i in agent_config.values()]).to(dev)

# environment parameters
env_config = {
    'n_agents' : sum(ag_count),
    'count' : ag_count,
    'time_steps': 24,
    'periods': np.array([1, 2]),

    'gen_capacity': np.repeat(gen_cap, ag_count),
    'gen_cost' : np.repeat(gen_cost, ag_count),
    'gen_impact': 0.5,

    'requirement': np.repeat(req_amt, ag_count),
    'penalty': 50,
    'penalty_type': 'diff',

    'price_start': 50,
    'friction': 2,
    'sigma': 3,
    
    'decay': 0,
    
    'zero_sum': False
}


# set seeds

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
                            dev = dev, zero_sum = env_config['zero_sum'])

obj = nash_dqn.nash_dqn(env,
                        n_agents = env_config['n_agents'], ag_count = env_config['count'],
                        gamma = config['gamma'], alpha=config['alpha'], beta=config['beta'],
                        lr = config['learning_rate'], trade_coef = 50, trade_soft = 0.25,
                        tau = config['tau'],
                        sched_step_size=config['sched_step_size'],
                        name = "test", n_nodes = config['n_nodes'], n_layers = config['n_layers'],
                        dev = dev)


obj.train(n_iter = 20000, 
          batch_size = 512, 
          n_plot = 5000,
          update_type = 'random')



#%%


obj.plot_nice()


#%% Big one

config={
        'random_seed': 2024,
        'learning_rate': 0.001,
        'gamma':1,
        'beta': 0,
        'alpha': 0.0,
        'tau':0.05,
        'sched_step_size': 25,
        'n_nodes': 150,
        'n_layers': 9,
    }

# agent setup
agent_config = {
    1 : {
        'gen_capacity': 3,
        'gen_cost': 150,
        'req': 40,
        'count': 2
        },
    2 : {
        'gen_capacity': 2.5,
        'gen_cost': 125,
        'req': 30,
        'count': 1
        },
    3 : {
        'gen_capacity': 2,
        'gen_cost': 100,
        'req': 30,
        'count': 1
        },
    
    4 : {
        'gen_capacity': 1.5,
        'gen_cost': 75,
        'req': 20,
        'count': 2
        },
    
    5  : {
        'gen_capacity': 1,
        'gen_cost': 50,
        'req': 10,
        'count': 3
        }
    
}

gen_cost =  torch.tensor([i['gen_cost'] for i in agent_config.values()]).to(dev)
gen_cap =  torch.tensor([i['gen_capacity'] for i in agent_config.values()]).to(dev)
req_amt =  torch.tensor([i['req'] for i in agent_config.values()]).to(dev)

ag_count = torch.tensor([i['count'] for i in agent_config.values()]).to(dev)

# environment parameters
env_config = {
    'n_agents' : sum(ag_count),
    'count' : ag_count,
    'time_steps': 24,
    'periods': np.array([1, 2]),

    'gen_capacity': np.repeat(gen_cap, ag_count),
    'gen_cost' : np.repeat(gen_cost, ag_count),
    'gen_impact': 0.25,

    'requirement': np.repeat(req_amt, ag_count),
    'penalty': 50,
    'penalty_type': 'diff',

    'price_start': 50,
    'friction': 5,
    'sigma': 3,
    
    'decay': 0,
    
    'zero_sum': False
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
                            dev = dev, zero_sum = env_config['zero_sum'])

obj = nash_dqn.nash_dqn(env,
                        n_agents = env_config['n_agents'], ag_count = env_config['count'],
                        gamma = config['gamma'], alpha=config['alpha'], beta=config['beta'],
                        lr = config['learning_rate'], trade_coef = 1000, trade_soft = 0.25,
                        tau = config['tau'],
                        sched_step_size=config['sched_step_size'],
                        name = "test", n_nodes = config['n_nodes'], n_layers = config['n_layers'],
                        dev = dev)


obj.train(n_iter = 40000, 
          batch_size = 512, 
          n_plot = 5000,
          update_type = 'random')



#%%


obj.plot_nice()

