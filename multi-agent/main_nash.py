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
import pandas as pd

#%%
if torch.cuda.is_available():  
    dev = torch.device("cuda:0")  
else:  
    dev = torch.device("cpu")    


#%%
# wandb.login()

def main(config_inject=dict()):
    config={
            'n_agents': 1,

            # hyperparametrs
            'random_seed': 3005,
            'learning_rate': 0.0002,
            'gamma': 0.9999,
            'tau':0.05,
            'sched_step_size': 50,
            'n_nodes': 36,
            'n_layers': 3,

            'epoch_scale': 200,
            'batch_size': 512,
            'n_plots': 1000,

            # parameters
            'T': 1/12,
            'S0':2.5,
            'sigma':0.5, 
            'kappa': 0.03, 
            'eta':0.05, 
            # generation capacity
            'xi': torch.tensor([2*0.5]).to(dev),
            # cost of generation
            'c':torch.tensor([2*1.25]).to(dev),
            # requirement
            'R':5, 
            # unit terminal penalty
            'pen':2.5,

            **config_inject
        }

    run = wandb.init(
        project='nash-dqn',
        entity='offset-credits',
        name = 'test_sweep',
        # hyperparameters and metadata
        config=config,
    )

    # set seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])


    #%%

    # xi and c have to be vectors of dimension n_agents

    N = 11

    gen_capacity = torch.tensor([2*0.5]).to(dev)
    cost = torch.tensor([2*1.25]).to(dev)

    env = offset_env.offset_env(
                                N = 11,
                                T=config['T'], S0=config['S0'], sigma=config['sigma'], 
                                kappa = config['kappa'], 
                                eta = config['eta'],  
                                xi = config['xi'], c = config['c'],  
                                R=config['R'], pen=config['pen'], 
                                n_agents=config['n_agents'],
                                penalty='diff',
                                dev=dev)

    obj = nash_dqn.nash_dqn(env,
                            n_agents=config['n_agents'],
                            gamma = config['gamma'], 
                            lr = config['learning_rate'],
                            tau = config['tau'],
                            sched_step_size=config['sched_step_size'],
                            name="test", n_nodes=config['n_nodes'], n_layers=config['n_layers'],
                            dev=dev)


    obj.train(n_iter=config['epoch_scale'] * N, 
              batch_size=config['batch_size'], 
              n_plot=config['n_plots'])

    # log performance 
    eval_fig, performance = obj.run_strategy(nsims=1000)
    trade_fig, gen_fig = obj.plot_policy()
    loss_fig = obj.loss_plots()

    eval_buf = io.BytesIO()
    trade_buf = io.BytesIO()
    gen_buf = io.BytesIO()
    loss_buf = io.BytesIO()

    eval_fig.savefig(eval_buf, format='png', bbox_inches='tight')
    trade_fig.savefig(trade_buf, format='png', bbox_inches='tight')
    gen_fig.savefig(gen_buf, format='png', bbox_inches='tight')
    loss_fig.savefig(loss_buf, format='png', bbox_inches='tight')

    eval_buf.seek(0)
    trade_buf.seek(0)
    gen_buf.seek(0)
    loss_buf.seek(0)

    wandb.log({'strategy_evaluation': wandb.Image(Image.open(eval_buf)),
               'trading_strategy': wandb.Image(Image.open(trade_buf)),
               'generation_strategy': wandb.Image(Image.open(gen_buf)),
               'training_loss': wandb.Image(Image.open(loss_buf)),
               **performance})

    dill.dump(obj, open(f'trained_{N}.pkl', "wb"))
    run.log_model(path='./'+ 'trained_' + str(N) + '.pkl', name=f'chkpt_{N}')

    wandb.finish()    


main()

# Testing
# hps = pd.read_csv('hp_test.csv')
# for i in range(len(hps)):
#     main(config_inject={
#         _col : hps.iloc[i][_col] for _col in hps.columns
#     })



# Hyperparameter Tuning 
# 2: Define the search space
# sweep_configuration = {
#     "method": "random",
#     "metric": {"goal": "maximize", "name": "cvar"},
#     "parameters": {
#         "learning_rate": {"max": 0.005, "min": 0.00001},
#         "sched_step_size": {"max": 0.005, "min": 0.00001},
#         "epoch_scale": {"min": 200, "max": 400},
#     },
# }
# sweep setup
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="nash-dqn", entity='offset-credits')
# print(sweep_id)
# wandb.agent(sweep_id, function=main, count=10)



# Load and Run Model
# from datetime import datetime
# with open('trained_100.pkl', 'rb') as in_strm:
#     ddpg_loaded = dill.load(in_strm)
#     ddpg_loaded.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
#     ddpg_loaded.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
