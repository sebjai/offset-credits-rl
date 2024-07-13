# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:41:23 2022

@author: sebja
"""

import numpy as np
import tqdm
import pdb
import torch


class offset_env():

    def __init__(self, T=1/12, sigma=0.5, kappa=0.03, eta = 0.05, xi=0.1,
                     c=0.25, S0=2.5, R=5, pen=2.5, N=50,
                     n_agents=2,
                     penalty='terminal', dev=torch.device("cpu")):
        
        self.dev = dev   
        #vector of period terminus'
        self.T=T
         # trading friction
        self.sigma = sigma
        self.kappa = kappa
        # impulse to generation 
        self.eta = eta
        # generation capacity
        self.xi = xi
        # cost of generation
        self.c = c
        self.S0 = S0
        # terminal inventory requirement
        self.R = R 
        # terminal penalty
        self.pen = pen
        # inventory and trade rate limits
        self.X_max = (1.2 * R) * self.T.size
        self.nu_max = 50
        
        self.n_agents=n_agents
        
        self.N = N
        
        self.t = np.linspace(0,self.T[-1], self.N * self.T.size + 1)
        self.dt = self.t[1]-self.t[0]  # time steps
        self.inv_vol = self.sigma * np.sqrt(0.5*self.T[0] )
        
        self.penalty = penalty
        self.diff_cost = lambda x0, x1 : self.pen * (  torch.maximum(self.R - x1, torch.tensor(0))\
                                            - torch.maximum(self.R - x0, torch.tensor(0)) ) 
            
        self.terminal_cost = lambda x0 : self.pen * torch.maximum ( self.R - x0, torch.tensor(0))
        
        self.term_excess = lambda x0, s0 :  - self.pen * torch.maximum (self.R - x0, torch.tensor(0)) \
                                            + torch.einsum('ij,i->ij', torch.maximum (x0 - self.R, torch.tensor(0)), s0)
        
    def randomize(self, batch_size=10, epsilon=0):
        # experiment with distributions
        # penalty + N(0,1)
        # S0 = self.S0 + 3*torch.randn(batch_size) * self.inv_vol 
        u = torch.rand(batch_size).to(self.dev)
        S0 = (self.S0 - 3*self.inv_vol) * (1-u) \
            + (self.S0 + 3*self.inv_vol) * u
        # Unifrom(0,x_max)
        X0 = torch.rand(batch_size, self.n_agents).to(self.dev) * self.X_max
        # randomized time 
        if self.penalty == 'diff':
            t0 = torch.tensor(np.random.choice(self.t[:-1], size=batch_size, replace=True)).float().to(self.dev)
        else:
            t0 = torch.tensor(np.random.choice(self.t[:-1], size=batch_size, replace=True)).float().to(self.dev)
            #TODO: figure out how to allow for this with multiple periods...
            #idx = (torch.rand(batch_size).to(self.dev) < epsilon)
            #t0[idx] = (self.T - self.dt)
        
        return t0, S0, X0
      
    def step(self, y, a, flag=1):
        
        #pdb.set_trace()
        
        batch_size = y.shape[0]
        
        # G = 1 is a generate a credit by investing in a project
        G = 1 * (a[:,1::2] > torch.rand(batch_size, self.n_agents).to(self.dev))
        
        yp = torch.zeros(y.shape).to(self.dev)
        
        # time evolution
        yp[:,0] = y[:,0] + self.dt
        
        # SDE step
        
        #pdb.set_trace()
        
        #TODO:
            #need to set so it converges to the proper T in the vector
            #time points are randomized
            
        count = 0
        
        #pdb.set_trace()
        
        T_list = torch.tensor(self.T).to(self.dev)
        
        # verify inclusive or exclusive inequality
        period = torch.tensor([min(self.T, key=lambda i:i if (i-x)>=0 else float('inf')) for x in y[:,0].detach().numpy()]).to(self.dev)
        
        
        eff_vol = self.sigma * torch.sqrt((self.dt * (period - yp[:,0]).clip(min = 0) / (period - y[:,0])))
        
        
        yp[:,1] = (y[:,1]- self.eta * torch.sum(self.xi * G,axis=-1)) *(period - yp[:,0]).clip(min = 0)/(period-y[:,0]) \
            + self.dt/(period-y[:,0]) * self.pen \
                + eff_vol  * torch.randn(batch_size).to(self.dev)
        

        # inventory evolution
        # nu = (1 - G) * a[:,::2] #-- assumes can only trade OR generate at any time, not both
        
        nu = a[:,::2] #-- allows to simultaneously trade and generate
        
        yp[:,2:] = y[:,2:] + self.xi * G + nu * self.dt
        
        # Reward
        
        #pdb.set_trace()
        
        if self.penalty == 'terminal':
            
                ind_T = (torch.abs(yp[:,0]-period)<1e-6).int()
            
            
            #if ind_T:
                r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                      + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                          + self.c * G \
                              + torch.einsum('ij,i->ij', self.terminal_cost(yp[:,2:]), ind_T) )
                             # + ind_T * self.terminal_cost(yp[:,2:]) )
                    
            #else:
               # r = -( y[:,1].reshape(-1,1) * nu *self.dt \
               #       + (0.5 * self.kappa * nu**2 * self.dt) * flag \
               #           + self.c * G )
            
                
        elif self.penalty == 'term_excess':
            
                ind_T = (torch.abs(yp[0,0]-period)<1e-6).int()
            
            #if ind_T:
                fut_price = (yp[:,1] + self.sigma * self.dt ** (1/2)) / ((1+0.5))
                    
                #terminal_cost = self.pen * torch.maximum(self.R - yp[:,2], torch.tensor(0))
                
                r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                      + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                          + self.c * G ) \
                            + torch.einsum('ij,i->ij', self.term_excess(yp[:,2:], fut_price), ind_T)
                           # + ind_T * self.term_excess(yp[:,2:], fut_price) 
           # else:
            
                #r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                 #     + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                 #         + self.c * G )
            
                
        elif self.penalty == 'diff':
            
            r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                  + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                      + self.c * G \
                          + self.diff_cost(y[:,2:], yp[:,2:]) )
        
        return yp, r
