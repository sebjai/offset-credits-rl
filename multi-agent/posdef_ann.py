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
                  activation='silu', env = None):
        super(posdef_ann, self).__init__()
        
        self.env = env
        
        self.prop_in_to_h = nn.Linear(n_in, nNodes)

        self.prop_h_to_h = nn.ModuleList(
            [nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])

        matrix_size = 2*n_agents
        self.prop_h_to_out = nn.Linear(nNodes, int(matrix_size*(matrix_size+1)/2))
        
        self.n_agents = n_agents
        
        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation == 'relu':
            self.g = nn.ReLU()
        elif activation == 'sigmoid':
            self.g= torch.sigmoid()
        
    def forward(self, x):
        
        pdb.set_trace()
        
        x_nrm = self.normalize(x, 'state')

        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x_nrm))

        for prop in self.prop_h_to_h:
            h = self.g(prop(h))

        # hidden layer to output layer
        y = self.prop_h_to_out(h)

        # generate the P matrix
        U = torch.zeros((x.shape[0], 2*self.n_agents, 2*self.n_agents))
        W = torch.zeros((x.shape[0], 2*self.n_agents, 2*self.n_agents))
        
        tril_indices = torch.tril_indices(row=2*self.n_agents, col=2*self.n_agents, offset=0)
        U[:,tril_indices[0], tril_indices[1]] = y  
        
        W[:,:2,:2] = U[:,:2,:2]
        
        # zero the top 2x2 portion of U
        U = U - W
        
        # symmetrize the remainder
        U = 0.5*(U + U.transpose(1,2))
        
        # generate a positive definite 2x2 embedded in the 2Kx2K
        
        I = torch.zeros(W.shape)
        rng = range(2)
        I[:,rng,rng] = 1
        
        W = torch.matmul(W, torch.transpose(W,1,2)) + 1e-5 * I
        
        P = W + U
            
        return P


    def norm(self, k: torch.tensor, typ :str):
        
        norm = torch.zeros(k.shape)
        
        if typ == 'state':
            norm[...,0] = self.env.T
            norm[...,1] = self.env.S0
            norm[...,2] = self.env.X_max
            
        if typ == 'policy':
            norm[...,0] = self.env.nu_max
            norm[...,1] = 1.0
        
        # norm = torch.ones(k.shape)
            
        return norm

    def normalize(self, k: torch.tensor, typ: str):
        '''
        possible types: "state" and "policy"
        '''
        norm = self.norm(k, typ)
            
        return k / norm

    def de_normalize(self, k: torch.tensor, typ: str):
        
        norm = self.norm(k, typ)
            
        return k * norm