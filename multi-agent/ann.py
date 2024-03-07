# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:29:21 2024

@author: sebja
"""

import torch
import torch.nn as nn

class ann(nn.Module):

    def __init__(self, n_in, n_out, nNodes, nLayers, 
                  activation='silu', out_activation=None, env=None):
        super(ann, self).__init__()
        
        self.prop_in_to_h = nn.Linear(n_in, nNodes)

        self.prop_h_to_h = nn.ModuleList(
            [nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])

        self.prop_h_to_out = nn.Linear(nNodes, n_out)
        
        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation == 'relu':
            self.g = nn.ReLU()
        elif activation == 'sigmoid':
            self.g= torch.sigmoid()
        
        self.out_activation = out_activation
        self.env = env

    def forward(self, x):
        
        x_nrm = self.normalize(x, 'state')

        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x_nrm))

        for prop in self.prop_h_to_h:
            h = self.g(prop(h))

        # hidden layer to output layer
        y = self.prop_h_to_out(h)

        if self.out_activation is not None:
            for i in range(y.shape[-1]):
                y[...,i] = self.out_activation[i](y[...,i])
            
        return y


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