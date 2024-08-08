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
        self.X_max = (1.5 * R) * self.T.size
        self.nu_max = 100
        
        self.n_agents=n_agents
        
        self.N = N
        
        self.t = np.linspace(0,self.T[-1], self.N * self.T.size + 1)
        self.dt = self.t[1]-self.t[0]  # time steps
        self.inv_vol = self.sigma * np.sqrt(0.5*self.T[0] )
        
        self.penalty = penalty
        
        # at terminal time, we don't want any excess...
        self.excess = lambda x0 : (self.pen * torch.maximum ( torch.subtract(x0, self.R), torch.tensor(0)) ) 
        
        self.diff_cost = lambda x0, x1, rem : rem * ( self.pen * (  torch.maximum(self.R - x1, torch.tensor(0))\
                                            - torch.maximum(self.R - x0, torch.tensor(0)) ) )
            
        self.terminal_cost = lambda x0 : self.pen * torch.maximum ( torch.subtract(self.R, x0), torch.tensor(0))
        
    def randomize(self, batch_size, epsilon):
        # experiment with distributions
        # penalty + N(0,1)
        # S0 = self.S0 + 3*torch.randn(batch_size) * self.inv_vol 
        
        #Try making it an equally spaced list for X and S?
        
        u = torch.rand(batch_size).to(self.dev)
        S0 = (self.S0 - 3*self.inv_vol) * (1-u) \
            + (self.S0 + 3*self.inv_vol) * u
        # Unifrom(0,x_max)
        X0 = (self.X_max + 2) * torch.rand(batch_size, self.n_agents).to(self.dev) - 2 # add a negative buffer
        
        #X0 = torch.rand(batch_size, self.n_agents).to(self.dev) * self.X_max
        # randomized time 
        if self.penalty == 'diff':
            t0 = torch.tensor(np.random.choice(self.t[:-1], size=batch_size, replace=True)).float().to(self.dev)
        else:
            t0 = torch.tensor(np.random.choice(self.t[:-1], size=batch_size, replace=True)).float().to(self.dev)
            for k in range(self.T.size):
                idx_i = (torch.rand(batch_size).to(self.dev) < epsilon / self.T.size ).int()
                t0[idx_i] = (self.T[k] - self.dt)
            
        return t0, S0, X0
    
    
      
    def step(self, y, a, flag, epsilon):
        
        #pdb.set_trace()
        
        batch_size = y.shape[0]
        
        # G = 1 is a generate a credit by investing in a project
        # random action (probability from the NN)
        G = 1 * (a[:,1::2] > torch.rand(batch_size, self.n_agents).to(self.dev))
        
        #binary outcome from the NN
        # G = a[:,1::2]
        
        yp = torch.zeros(y.shape).to(self.dev)
        
        # time evolution
        yp[:,0] = y[:,0] + self.dt
        
        # SDE step
        
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
        
        if self.penalty == 'terminal':
            
            ind_T = (torch.abs(yp[:,0]-period)<1e-6).int()
            
            r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                  + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                      + self.c * G \
                          + torch.einsum('ij,i->ij', self.terminal_cost(yp[:,2:]), ind_T) )
                
            yp[:,2:] = yp[:,2:] - torch.einsum('ij,i->ij', torch.min( yp[:,2:] , self.R ), ind_T) 
           
               
        elif self.penalty == 'excess':
            
            ind_T = (torch.abs(yp[:,0]-period)<1e-6).int()
            
            end = (torch.abs(yp[:,0]-self.T[-1])<1e-6).int()
            
            decay = epsilon * 2
            
            r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                  + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                      + self.c * G \
                          + torch.einsum('ij,i->ij', self.terminal_cost(yp[:,2:]), ind_T) \
                              + decay * torch.einsum('ij,i->ij', self.excess(yp[:,2:]), end) * flag)
                
            yp[:,2:] = yp[:,2:] - torch.einsum('ij,i->ij', torch.min( yp[:,2:] , self.R ), ind_T) 
           
        
        elif self.penalty == 'diff':
            
            ind_T = (torch.abs(yp[:,0]-period)<1e-6).int()
            
            remain = self.T.size - torch.tensor([((torch.tensor(self.T) == i)).nonzero()[0] for i in period])

            
            r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                  + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                      + self.c * G \
                          + self.diff_cost(y[:,2:], yp[:,2:], remain) )
                
            
            yp[:,2:] = yp[:,2:] - torch.einsum('ij,i->ij', torch.min( yp[:,2:] , self.R ), ind_T) 
        
        return yp, r
