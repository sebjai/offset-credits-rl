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
        'beta': 50,
        'alpha': 0,
        'tau':0.05,
        'sched_step_size': 25,
        'n_nodes': 150,
        'n_layers': 3,

    }

# set seeds

torch.manual_seed(config['random_seed'])
np.random.seed(config['random_seed'])


# xi and c have to be vectors of dimension n_agents

n_agents = 4

gen_capacity = torch.tensor([1, 0.5, 0.25, 0.5]).to(dev)
cost = torch.tensor([2.5, 1.25, 0.625, 1.25]).to(dev)

periods = np.array([1/12, 2/12])

#gen_capacity = torch.tensor([ 0.25, 0.5, 0.25, 0.5, 0.25]).to(dev)
#cost = torch.tensor([0.625, 1.25, 0.625, 1.25, 0.625]).to(dev)

# either is one element or n_agents length, allows for different requirements
#Req = torch.tensor([5, 4, 4, 3, 3]).to(dev)
Req = torch.tensor([5, 5, 4, 4])

env = offset_env.offset_env(T=periods, S0=2.5, sigma=0.25, 
                            kappa = 0.5, 
                            eta = 0.05, 
                            xi = gen_capacity, c = cost,  
                            R=Req, pen=2.5, 
                            n_agents=n_agents,
                            N = 25,
                            penalty='excess',
                            dev=dev)

obj = nash_dqn.nash_dqn(env,
                        n_agents=n_agents,
                        gamma = config['gamma'], beta = config['beta'], alpha = config['alpha'],
                        lr = config['learning_rate'],
                        tau = config['tau'],
                        sched_step_size=config['sched_step_size'],
                        name="test", n_nodes=config['n_nodes'], n_layers=config['n_layers'],
                        dev=dev)


obj.train(n_iter= 5000, 
         batch_size=2048, 
          n_plot=5000,
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
 
# =============================================================================
# n_agents = 2
# 
# gen_capacity = torch.tensor([0.2, 0.4]).to(dev)
# cost = torch.tensor([0.5, 1.0]).to(dev)
# 
# env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.5, 
#                             kappa = 0.15, 
#                             eta = 0.05, 
#                             xi = gen_capacity, c = cost,  
#                             R=5, pen=2.5, 
#                             n_agents=n_agents,
#                             N = 26,
#                             penalty='diff')
# 
# obj = nash_dqn.nash_dqn(env,
#                         n_agents=n_agents,
#                         gamma = config['gamma'], 
#                         lr = config['learning_rate'],
#                         tau = config['tau'],
#                         sched_step_size=config['sched_step_size'],
#                         name="test", n_nodes=config['n_nodes'], n_layers=config['n_layers'])
# 
# 
# 
# obj.train(n_iter=3000, 
#           batch_size=512, 
#           n_plot=1000) 
#  
#  
#  
#  
# =============================================================================
 
 #%%
# =============================================================================
#  
# n_agents = 4
# 
# gen_capacity = torch.tensor([0.2, 0.2, 0.4, 0.4]).to(dev)
# cost = torch.tensor([0.5, 0.5, 1, 1]).to(dev)
# 
# env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.5, 
#                             kappa = 0.03, 
#                             eta = 0.05, 
#                             xi = gen_capacity, c = cost,  
#                             R=5, pen=2.5, 
#                             n_agents=n_agents,
#                             N = 50,
#                             penalty='diff')
# 
# obj = nash_dqn.nash_dqn(env,
#                         n_agents=n_agents,
#                         gamma = config['gamma'], 
#                         lr = config['learning_rate'],
#                         tau = config['tau'],
#                         sched_step_size=config['sched_step_size'],
#                         name="test", n_nodes=config['n_nodes'], n_layers=config['n_layers'])
# 
# 
# 
# obj.train(n_iter=1_000, 
#           batch_size=256, 
#           n_plot=100)  
#  
#  
#  
#  
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
