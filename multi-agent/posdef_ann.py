# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:12:51 2024

@author: sebja
"""

import torch
import torch.nn as nn
import pdb

class posdef_ann(nn.Module):

    def __init__(self, n_in, n_agents, nNodes, nLayers, 
                  activation='silu', env = None, dev=torch.device("cpu")):
        super(posdef_ann, self).__init__()
        
        self.dev = dev
                
        self.env = env
        
        self.prop_in_to_h = nn.Linear(n_in, nNodes)

        self.prop_h_to_h = nn.ModuleList(
            [nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])

        self.prop_h_to_out = nn.Linear(nNodes, int(3+4*n_agents*(n_agents-1)))
        
        self.n_agents = n_agents
        
        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation == 'relu':
            self.g = nn.ReLU()
        elif activation == 'sigmoid':
            self.g= torch.sigmoid()
        
    def forward(self, x):
        
        x_nrm = self.normalize(x)

        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x_nrm))

        for prop in self.prop_h_to_h:
            h = self.g(prop(h))

        # hidden layer to output layer
        y = self.prop_h_to_out(h)
        
        # generate the P matrix
        U = torch.zeros((x.shape[0], 2*self.n_agents, 2*self.n_agents)).to(self.dev)
        V = torch.zeros((x.shape[0], 2*self.n_agents, 2*self.n_agents)).to(self.dev)
        W = torch.zeros((x.shape[0], 2*self.n_agents, 2*self.n_agents)).to(self.dev)
        
        # generate a positive definite 2x2 embedded in the 2Kx2K
        W[:,0,0] = y[:,0]
        W[:,0,1] = y[:,1]
        W[:,1,1] = y[:,2]
        
        I = torch.zeros(W.shape).to(self.dev)
        I[:,0,0] = 1
        I[:,1,1] = 1
        
        W = torch.matmul(W, torch.transpose(W,1,2)) + 1e-5 * I
                
        
        # fill the 2 x 2(n-1) top right block of U and symmetrize
        U[:,:2,2:] = y[:,3:3+4*(self.n_agents-1)].reshape(y.shape[0],2,-1)
        U = U + U.transpose(1,2)
        
        # fill the 2(n-1) x 2(n-1) lower right block
        V[:,2:,2:] = y[:,3+4*(self.n_agents-1):].reshape(y.shape[0],2*(self.n_agents-1),2*(self.n_agents-1))
        V = V + V.transpose(1,2)
        
        
        P = U + V + W
        
        return P
    
    def norm(self, y: torch.tensor):
        
        norm = torch.zeros(y.shape).to(self.dev)
        
        norm[...,0] = self.env.T[-1]
        norm[...,1] = self.env.S0
        norm[...,2:] = self.env.X_max
        
        # norm = torch.ones(k.shape)
            
        return norm

    def normalize(self, y: torch.tensor):
        
        norm = self.norm(y)
            
        return y / norm

    def de_normalize(self, k: torch.tensor, typ: str):
        
        norm = self.norm(k, typ)
            
        return k * norm    