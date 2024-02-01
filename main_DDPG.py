# -*- coding: utf-8 -*-

# import os, sys
# sys.path.insert(0, os.path.abspath(".."))

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

import DDPG, offset_env
import dill

#%%
# =============================================================================
# env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.5, 
#                             kappa = 0.03, 
#                             eta = 0.05, xi=0.1, c=0.25,  
#                             R=5, pen=2.5, 
#                             N = 101,
#                             penalty='diff')
# 
# ddpg = DDPG.DDPG(env,
#             gamma = 1, 
#             lr = 0.001,
#             tau = 0.005,
#             sched_step_size=50,
#             name="test", n_nodes=64, n_layers=5)
# =============================================================================
 
#%%    
# try some transfer learning -- learn with various values of N
# starting from the optimal from the previous value of N

# =============================================================================
# 
# for N in [10, 25, 50, 100]:
#     
#     print("\n***************************************")
#     print("N=" + str(N))
#     
#     scale = 1 #(100/N)
# 
#     env = offset_env.offset_env(T=1/12, S0=2.5, sigma=0.5, 
#                                 kappa = 0.03, 
#                                 eta = 0.05, 
#                                 xi=0.1*scale, c=scale*0.25,  
#                                 R=5, pen=2.5, 
#                                 N = N,
#                                 penalty='diff')
#     
#     ddpg.reset(env)    
#     
#     ddpg.train(n_iter = 5_000, 
#                n_plot = 2500, 
#                batch_size = 256, 
#                n_iter_Q=5, n_iter_pi=1)
#     
#     #dill.dump(ddpg, open('trained_' + str(N) + '.pkl', "wb"))
#     
#     
# =============================================================================
#%%


# =============================================================================
# 
# ddpg_trained = dill.load(open('trained_100.pkl', 'rb'))
# ddpg_trained.run_strategy()
# ddpg_trained.plot_policy()
# 
# 
# =============================================================================
#ddpg_trained.train(n_iter = 50000, n_plot = 25000)


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
            tau = 0.005,
            sched_step_size=50,
            name="test", n_nodes=64, n_layers=5)

dill.dump(ddpg, open('base' + '.pkl', "wb"))

#%%

ddpg_base1 = dill.load(open('base.pkl', 'rb'))

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
    
    ddpg_base1.reset(env)   
    
    ddpg_base1.train(n_iter = 7500, 
               n_plot = 7500, 
               batch_size = 256, 
               n_iter_Q=5, n_iter_pi=1)



#%%


ddpg_base2 = dill.load(open('base.pkl', 'rb'))

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
    
<<<<<<< HEAD
    ddpg_base2.reset(env)   
    
    ddpg_base2.train(n_iter = 7500, 
               n_plot = 7500, 
               batch_size = 256, 
               n_iter_Q=5, n_iter_pi=1)


#%%


ddpg_base3 = dill.load(open('base.pkl', 'rb'))

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
    
    ddpg_base3.reset(env)   
    
    ddpg_base3.train(n_iter = 7500, 
               n_plot = 7500, 
               batch_size = 256, 
               n_iter_Q=5, n_iter_pi=1)




#%%

ddpg_base1.run_strategy()
ddpg_base2.run_strategy()
ddpg_base3.run_strategy()

ddpg_base1.plot_policy()
ddpg_base2.plot_policy()
ddpg_base3.plot_policy()








=======

# from datetime import datetime
# with open('trained_100.pkl', 'rb') as in_strm:
#     ddpg_loaded = dill.load(in_strm)
#     ddpg_loaded.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
#     ddpg_loaded.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
>>>>>>> f3d72ff9c80e3ec4fc80f801c3e50cfbf3e87b6f
