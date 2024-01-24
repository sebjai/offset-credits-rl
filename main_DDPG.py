# -*- coding: utf-8 -*-

# import os, sys
# sys.path.insert(0, os.path.abspath(".."))

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

import DDPG, offset_env
import dill

#%%
env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.5, 
                            kappa = 0.03, 
                            eta = 0.05, xi=0.1, c=0.25,  
                            R=5, pen=2.5, 
                            N = 101,
                            penalty='diff')

ddpg = DDPG.DDPG(env,
            gamma = 1, 
            lr = 0.001,
            tau = 0.001,
            sched_step_size=50,
            name="test", n_nodes=64, n_layers=5 )
 
#%%    
# try some transfer learning -- learn with various values of N
# starting from the optimal from the previous value of N
for N in [10, 25, 50, 100]:
    
    print("\n***************************************")
    print("N=" + str(N))
    
    scale = 1 #(100/N)

    env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.5, 
                                kappa = 0.03, 
                                eta = 0.05, 
                                xi=0.1*scale, c=scale*0.25,  
                                R=5, pen=2.5, 
                                N = N,
                                penalty='diff')
    
    ddpg.reset(env)    
    
    ddpg.train(n_iter = 5_000, 
               n_plot = 1000, 
               batch_size = 128, 
               n_iter_Q=5, n_iter_pi=1)
    
    dill.dump(ddpg, open('trained_' + str(N) + '.pkl', "wb"))
    
    
