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
                     c=0.25, S0=2.5, R=5, pen=2.5, N=51,
                     n_agents=2,
                     penalty='terminal'):
        
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
        self.X_max = 1.5 * R
        self.nu_max = 100.0
        
        self.n_agents=n_agents
        
        self.N = N
        self.t = np.linspace(0,self.T, self.N)
        self.dt = self.t[1]-self.t[0]  # time steps
        self.inv_vol = self.sigma * np.sqrt(0.5*self.T )
        
        self.penalty = penalty
        self.diff_cost = lambda x0, x1 : self.pen * (  torch.maximum(self.R - x1, torch.tensor(0))\
                                            - torch.maximum(self.R - x0, torch.tensor(0)) )        
        
    def randomize(self, batch_size=10, epsilon=0):
        # experiment with distributions
        # penalty + N(0,1)
        # S0 = self.S0 + 3*torch.randn(batch_size) * self.inv_vol 
        u = torch.rand(batch_size)
        S0 = (self.S0 - 3*self.inv_vol) * (1-u) \
            + (self.S0 + 3*self.inv_vol) * u
        # Unifrom(0,x_max)
        X0 = torch.rand(batch_size, self.n_agents) * self.X_max
        # randomized time 
        t0 = torch.tensor(np.random.choice(self.t[:-1], size=batch_size, replace=True)).to(torch.float32)
        idx = (torch.rand(batch_size) < epsilon)
        t0[idx] = (self.T - self.dt)
        
        return t0, S0, X0
      
    def step(self, y, a, flag=1):
        
        batch_size = y.shape[0]
        
        # G = 1 is a generate a credit by investing in a project
        G = 1 * (a[:,1::2] > torch.rand(batch_size, self.n_agents))
        
        yp = torch.zeros(y.shape)
        
        # time evolution
        yp[:,0] = y[:,0] + self.dt
        
        # SDE step
        eff_vol = self.sigma * torch.sqrt((self.dt * (self.T - yp[:,0]) / (self.T - y[:,0])))
        
        yp[:,1] = (y[:,1]- self.eta * self.xi * torch.sum(G,axis=-1)) *(self.T - yp[:,0])/(self.T-y[:,0]) \
            + self.dt/(self.T-y[:,0]) * self.pen \
                + eff_vol  * torch.randn(batch_size)
                            
        # inventory evolution
        nu = (1-G) * a[:,::2]
        yp[:,2:] = y[:,2:] + self.xi * G + nu * self.dt
        
        # Reward
        if self.penalty == 'terminal':
            
            ind_T = (torch.abs(yp[:,0]-self.T)<1e-6).int()
            # terminal_cost = self.pen * torch.maximum(self.R - yp[:,2], torch.tensor(0))
            
            # r = -( y[:,1] * nu *self.dt \
            #       + (0.5 * self.kappa * nu**2 * self.dt) * flag \
            #           + self.c * G \
            #               + ind_T * terminal_cost)
                
        elif self.penalty == 'diff':
            
            r = -( y[:,1].reshape(-1,1) * nu *self.dt \
                  + (0.5 * self.kappa * nu**2 * self.dt) * flag \
                      + self.c * G \
                          + self.diff_cost(y[:,2:], yp[:,2:]) )
        
        return yp, r
