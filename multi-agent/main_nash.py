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


#%%
config={
        'random_seed': 2024,
        'learning_rate': 0.005,
        'gamma': 1,
        'beta': 0,
        'alpha': 0,
        'tau':0.05,
        'sched_step_size': 25,
        'n_nodes': 150,
        'n_layers': 3,
    }

# agent setup
agent_config = {
    1 : {
        'gen_capacity': 1,
        'gen_cost': 2.5
        },
    # 2 : {
    #     'gen_capacity': 1,
    #     'gen_cost': 2.5
    #     },
    3  : {
        'gen_capacity': 0.75,
        'gen_cost': 1.875
        },
    # 4 : {
    #     'gen_capacity': 0.5,
    #     'gen_cost': 1.25
    #     },
    5 : {
        'gen_capacity': 0.25,
        'gen_cost': 0.625
        },
    # 6 : {
    #     'gen_capacity': 0.25,
    #     'gen_cost': 0.625
    #     }
}

# environment parameters
env_config = {
    'n_agents' : 3,
    'time_steps': 10,
    'periods': np.array([1/12, 2/12]),

    'gen_capacity': torch.tensor([i['gen_capacity'] for i in agent_config.values()]).to(dev),
    'gen_cost' : torch.tensor([i['gen_cost'] for i in agent_config.values()]).to(dev),
    'gen_impact': 0.05,

    'requirement': torch.tensor([
                                 5, 
                                 4,  
                                 3,
                                 ]).to(dev),
    'penalty': 2.5,
    'penalty_type': 'diff',

    'price_start': 2.5,
    'friction': 0.3,
    'sigma': 0.25,
    
    'decay': 0
}

assert(len(env_config['requirement']) == env_config['n_agents'])
assert(len(agent_config) == env_config['n_agents'])

# run = wandb.init(
#     project='offset credit rl',
#     # entity='offset-credits',
#     # name = 'base',
#     # Track hyperparameters and run metadata
#     config=config,
# )

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
                            dev = dev)

obj = nash_dqn.nash_dqn(env,
                        n_agents = env_config['n_agents'],
                        gamma = config['gamma'], alpha=config['alpha'], beta=config['beta'],
                        lr = config['learning_rate'],
                        tau = config['tau'],
                        sched_step_size=config['sched_step_size'],
                        name = "test", n_nodes = config['n_nodes'], n_layers = config['n_layers'],
                        dev = dev)


obj.train(n_iter = 10_000, 
          batch_size = 512, 
          n_plot = 5_000,
          update_type = 'random')



#%%

# =============================================================================
# for i in range(4):
#     
#     iters = np.array([10000, 5000, 5000, 5000])
# 
#     obj.train(n_iter= iters[i], 
#               batch_size=2048, 
#               n_plot=iters[i],
#               update_type = 'random')
# 
#     obj.reset(env = env)
# 
# =============================================================================
#%%

# =============================================================================
# env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.5, 
#                             kappa = 1, 
#                             eta = 0.05, 
#                             xi = gen_capacity, c = cost,  
#                             R=5, pen=2.5, 
#                             n_agents=n_agents,
#                             N = 26,
#                             penalty='diff',
#                             dev=dev)
# 
#  
# =============================================================================
# # =============================================================================
# 
# for kappas in [1.5, 1, 0.5, 0.25]:
# 
#     print("\n***************************************")
#     print("kappa=" + str(kappas))
#     
#     scale = 1 #(100/N)
# 
#     env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.25, 
#                                 kappa = kappas, 
#                                 eta = 0.05, 
#                                 xi = gen_capacity, c = cost,  
#                                 R=5, pen=2.5, 
#                                 n_agents=n_agents,
#                                 N = 26,
#                                 penalty='diff',
#                                 dev=dev)
#     
#     obj.reset(env)    
#     
#     obj = nash_dqn.nash_dqn(env,
#                             n_agents=n_agents,
#                             gamma = config['gamma'], 
#                             lr = config['learning_rate'],
#                             tau = config['tau'],
#                             sched_step_size=config['sched_step_size'],
#                             name="test", n_nodes=config['n_nodes'], n_layers=config['n_layers'],
#                             dev=dev)
#     
#     obj.train(n_iter=1000, 
#               batch_size=512, 
#               n_plot=1000)
# 
#     # log performance
#    
# dill.dump(obj, open('trained_kappa' + '.pkl', "wb"))
# =============================================================================

 
#%%    
# # try some transfer learning -- learn with various values of N
# # starting from the optimal from the previous value of N
# for N in [10, 25, 50, 100]:

#     print("\n***************************************")
#     print("N=" + str(N))
    
#     wandb.config.update({'N': N}, allow_val_change=True)
#     wandb.config.update({'global_epochs': config['epoch_scale'] * N}, allow_val_change=True)
#     scale = 1 #(100/N)

#     env = offset_env.offset_env(T=config['T'], S0=config['S0'], sigma=config['sigma'], 
#                                 kappa = config['kappa'], 
#                                 eta = config['eta'], 
#                                 xi=config['xi']*scale, c=scale*config['c'],  
#                                 R=config['R'], pen=config['pen'], 
#                                 N = N,
#                                 penalty='diff')
    
#     ddpg.reset(env)    
    
#     ddpg.train(n_iter = config['epoch_scale'] * N, 
#                n_plot = config['n_plots'], 
#                batch_size = config['batch_size'], 
#                n_iter_Q=config['Q_epochs'], n_iter_pi=config['pi_epochs'])

#     # log performance 
#     eval_fig, performance = ddpg.run_strategy(nsims=1000)
#     trade_fig, gen_fig = ddpg.plot_policy()

#     eval_buf = io.BytesIO()
#     trade_buf = io.BytesIO()
#     gen_buf = io.BytesIO()
    
#     eval_fig.savefig(eval_buf, format='png', bbox_inches='tight')
#     trade_fig.savefig(trade_buf, format='png', bbox_inches='tight')
#     gen_fig.savefig(gen_buf, format='png', bbox_inches='tight')

#     eval_buf.seek(0)
#     trade_buf.seek(0)
#     gen_buf.seek(0)

#     wandb.log({'strategy_evaluation': wandb.Image(Image.open(eval_buf)),
#                'trading_strategy': wandb.Image(Image.open(trade_buf)),
#                'generation_strategy': wandb.Image(Image.open(gen_buf)),
#                **performance})

    
#     dill.dump(ddpg, open('trained_' + str(N) + '.pkl', "wb"))
#     run.log_model(path='./'+ 'trained_' + str(N) + '.pkl', name=f'chkpt_{N}')

# wandb.finish()
    

# from datetime import datetime
# with open('trained_100.pkl', 'rb') as in_strm:
#     ddpg_loaded = dill.load(in_strm)
#     ddpg_loaded.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
#     ddpg_loaded.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
