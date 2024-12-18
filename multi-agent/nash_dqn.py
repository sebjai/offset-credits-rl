# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from offset_env import offset_env as Environment

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#from matplotlib import mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

import torch
import torch.optim as optim
import torch.nn as nn
# torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm
from ann import ann
from posdef_ann import posdef_ann

from replay_buffer import replay_buffer as rb

import copy

import pdb

from datetime import datetime
import wandb

class nash_dqn():

    def __init__(self, 
                 env: Environment,  
                 n_agents=2, ag_count = torch.tensor([1,1]),
                 gamma=0.9999, beta = 100, alpha = 0, trade_coef = 0, trade_soft = 0.5,
                 n_nodes=36, n_layers=3, 
                 lr=0.001, tau=0.005, sched_step_size = 20,
                 name="", dev=torch.device("cpu")):
    
        self.dev = dev

        self.env = env
        self.n_agents = n_agents
        self.gamma = gamma
        self.beta = beta
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name =  name
        self.sched_step_size = sched_step_size
        self.lr = lr
        self.tau = tau
        self.trade_coef = trade_coef
        self.trade_soft = trade_soft
        
        
        self.ag_count = ag_count
        
        
        self.alpha = alpha
        
        self.__initialize_NNs__()
        
        self.t = []
        self.S = []
        self.X = []
        self.nu = []
        self.p = []
        self.r = []
        self.epsilon = []
        
        
        self.VA_loss = []
        
    def reset(self, env):
        
        self.epsilon = []
        self.env = env
        self.lr = self.lr
        
        
        for g in range(self.n_agents):
            self.P[g]['net'].env = env
            self.psi[g]['net'].env = env
            self.V_main[g]['net'].env = env
            self.mu[g]['net'].env = env
            self.V_target[g]['net'].env = env
            
            for k in self.P[g]['optimizer'].param_groups:
                k['lr'] = self.lr
                
            for k in self.psi[g]['optimizer'].param_groups:
                k['lr'] = self.lr
            
            for k in self.V_main[g]['optimizer'].param_groups:
                k['lr'] = self.lr
            
            for k in self.mu[g]['optimizer'].param_groups:
                k['lr'] = self.lr
            
        
    def __initialize_NNs__(self):
        
        
        def create_net(n_in, n_out, n_nodes, n_layers, out_activation = None):
            net = ann(n_in, n_out, n_nodes, n_layers, 
                      out_activation = out_activation,
                      env=self.env, dev=self.dev).to(self.dev)
            
            optimizer, scheduler = self.__get_optim_sched__(net)
            
            return {'net' : net, 'optimizer' : optimizer, 'scheduler' : scheduler}
        
        def create_posdef_net(n_in, n_agents, n_nodes, n_layers):
            net = posdef_ann(n_in, n_agents, n_nodes, n_layers, env=self.env, dev=self.dev).to(self.dev)
            optimizer, scheduler = self.__get_optim_sched__(net)
            
            result = {'net' : net, 'optimizer' : optimizer, 'scheduler' : scheduler}
            
            return result
        
        # value network
        #   features are t, S,X
        #   output = value
        
        # policy approximation for mu =( nu and prob)
        #   features are t, S,X
        #   output = (rates_k, prob_k) k = 1,..K
        
        # shift ann psi
        #   features are t, S,X
        #   output = batch x (K-1)
        
        self.P = []
        
        self.psi = []
        
        self.V_main = []
        
        self.mu = []
        
        self.V_target = []
        
# =============================================================================
#         for k in range(self.n_agents):
#             self.V_main.append(create_net(n_in = (2 + self.n_agents), n_out = 1, n_nodes=32, n_layers=3))
#             
#             # try binary output activation instead of a probability.... might make graphs ugly but lets see
#             self.mu.append(create_net(n_in = (2 + self.n_agents), n_out = 2, n_nodes=32, n_layers=3,
#                                  out_activation=[lambda x : self.env.nu_max / 2 * torch.tanh(x),
#                                                  lambda x : (torch.sigmoid(x))]))
#             
#             self.V_target.append(copy.copy(self.V_main[k]))
#             
#             self.psi.append(create_net(n_in=(2 + self.n_agents), n_out= (2 * (self.n_agents - 1)), n_nodes=32, n_layers=3))
#             
#             self.P.append(create_posdef_net(n_in=(2 + self.n_agents), n_agents=self.n_agents, n_nodes=32, n_layers=3))
#             
#         
# =============================================================================

        for k in range(self.ag_count.size()[0]):
            
            self.V_main.append(create_net(n_in = (2 + self.n_agents), n_out = 1, n_nodes=32, n_layers=3))
            
            # try binary output activation instead of a probability.... might make graphs ugly but lets see
            self.mu.append(create_net(n_in = (2 + self.n_agents), n_out = 2, n_nodes=32, n_layers=3,
                                 out_activation=[lambda x : self.env.nu_max / 2 * torch.tanh(x),
                                                 lambda x : (torch.sigmoid(x))]))
            
            self.V_target.append(copy.copy(self.V_main[k]))
            
            self.psi.append(create_net(n_in=(2 + self.n_agents), n_out= (2 * (self.n_agents - 1)), n_nodes=32, n_layers=3))
            
            self.P.append(create_posdef_net(n_in=(2 + self.n_agents), n_agents=self.n_agents, n_nodes=32, n_layers=3))
            
            
            if self.ag_count[k] > 1:
                
                for j in range(self.ag_count[k] - 1):
                    #copy the most recent NN if the agent count is greater than 1
                    self.V_main.append(copy.copy(self.V_main[-1]))
                    
                    self.mu.append(copy.copy(self.mu[-1]))
                    
                    self.V_target.append(copy.copy(self.V_target[k]))
                    
                    self.psi.append(copy.copy(self.psi[-1]))
                    
                    self.P.append(copy.copy(self.P[-1]))
                    
               
    def __get_optim_sched__(self, net):
        
        optimizer = optim.AdamW(net.parameters(),
                                lr=self.lr)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.sched_step_size, gamma=0.999)
       
        return optimizer, scheduler
    
    def soft_update(self, main, target):
    
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def __stack_state__(self, t, S, X, plot1 = False):
        
        # normalization happens outside of stack state
        if plot1 == True:
            tSX = torch.cat((t.unsqueeze(-1), 
                            S.unsqueeze(-1),
                            X.unsqueeze(-1)), axis=-1)
        else:
            tSX = torch.cat((t.unsqueeze(-1), 
                             S.unsqueeze(-1),
                             X), axis=-1)
        
        return tSX
    
    
    def __grab_mini_batch__(self, batch_size, epsilon):
        
        t, S, X = self.env.randomize(batch_size = batch_size, epsilon = epsilon)
        
        return t, S, X
   
    def range_test(self, x, test='prob'):
        if test=='prob':
            if torch.amin(x) < 0 or torch.amax(x) > 1:
                print(torch.amin(x), torch.amax(x))
                
    def get_actions(self, Y, batch_size):
        
        MU = torch.zeros([batch_size, 2 * self.n_agents]).to(self.dev)
            
        for k in range(self.n_agents):
                
            MU[...,(2*k):((2*k)+2)] = self.mu[k]['net'](Y)
                
        return MU
            
    def get_value(self, Y, batch_size):
        
        Val = torch.zeros([batch_size, self.n_agents]).to(self.dev)
        
        for k in range(self.n_agents):
            
            Val[...,k:(k+1)] = self.V_main[k]['net'](Y)
            
        return Val
            
    def get_target(self, Y, batch_size):
        
        targ = torch.zeros([batch_size, self.n_agents]).to(self.dev)
        
        for k in range(self.n_agents):
            
            targ[...,k:(k+1)] = self.V_target[k]['net'](Y)
            
        return targ
            
    def get_value_advantage_mu(self, Y, Yp, batch_size):
        
        # Y = (t, S, X_1,..., X_K)
        
        MU = self.get_actions(Y, batch_size)
        MUp = self.get_actions(Yp, batch_size)
        
        V = self.get_value(Y, batch_size)
        Vp = self.get_target(Yp, batch_size)
        
        
        P  = []
        psi = []
        
        P_p = []
        psi_p = []
        
        for k in range(self.n_agents):
            P.append(self.P[k]['net'](Y))
            psi.append(self.psi[k]['net'](Y))
            
            P_p.append(self.P[k]['net'](Yp))
            psi_p.append(self.psi[k]['net'](Yp))
            
        mu = self.reorder_actions(MU)
        mu_p = self.reorder_actions(MUp)
            
        return mu, mu_p, V, Vp, P, P_p, psi, psi_p
            
    def reorder_actions(self, MU):
        
        mu =[]
        for k in range(self.n_agents):
            mu.append(torch.zeros(MU.shape).to(self.dev))
            
            idx = torch.ones(MU.shape[1]).bool().to(self.dev)
            idx[2*k:2*k+2] = False
            
            mu[k][:,:2] = MU[:,2*k:2*k+2]
            mu[k][:,2:] = MU[:,idx]
        
        return mu
        
    def zero_grad(self):
        
        for k in range(self.n_agents):
            self.P[k]['optimizer'].zero_grad()
            self.psi[k]['optimizer'].zero_grad()
            
            self.mu[k]['optimizer'].zero_grad()
            self.V_main[k]['optimizer'].zero_grad()        
        
    def step_optim(self, net):
        net['optimizer'].step()
        net['scheduler'].step()
    
    def CVaR(self, data, confidence_level = 0.95):
        # Set the desired confidence level
        signal = sorted(data)
        cvar_index = int((1 - confidence_level) * len(signal))
        cvar = np.mean(signal[:cvar_index])
        return cvar
    
    def randomize_actions(self, actions, epsilon):
        
        rate_idx = torch.arange(0, (2*self.n_agents), 2).to(self.dev)
        prob_idx = torch.arange(1, (2*self.n_agents), 2).to(self.dev)
        
        actions[:, (rate_idx)] += 0.2*self.env.nu_max * epsilon * torch.randn(actions[:, (rate_idx)].shape).to(self.dev)
        actions[:, (rate_idx)] = torch.clip(actions[:, (rate_idx)], min = -self.env.nu_max, max = self.env.nu_max)
        
        #pdb.set_trace()
        
        actions[:, (prob_idx)] += 0.5*epsilon * torch.randn(actions[:, (prob_idx)].shape).to(self.dev)
        actions[:, (prob_idx)] = torch.clip(actions[:, (prob_idx)], min = 0, max = 1)
        
        return actions
    
    
    def freeze_net(self, net):
        
        for param in net.parameters():
            param.requires_grad = False 
            
    def unfreeze_net(self, net):
        
        for param in net.parameters():
            param.requires_grad = True 
        
    
    def update_nets(self, update):
        
        if update =='V':
            for k in range(self.n_agents):
                self.step_optim(self.V_main[k])
            
        elif update == 'mu':
            for k in range(self.n_agents):
                self.step_optim(self.mu[k])
            
        elif update == 'A':
            for k in range(self.n_agents):
                self.step_optim(self.P[k])
                self.step_optim(self.psi[k])
                
        elif update =='all':
            
            #self.step_optim(self.V_main)
            #self.step_optim(self.mu)
# =============================================================================
#             
#             for k in range(self.n_agents):
#                 
#                 self.step_optim(self.mu[k])
#                 self.step_optim(self.V_main[k])
#                 self.step_optim(self.P[k])
#                 self.step_optim(self.psi[k])
#                 
# =============================================================================
                
            ii = 0
            
            for k in range(self.ag_count.size()[0]):
                
            
                self.step_optim(self.mu[ii])
                self.step_optim(self.V_main[ii])
                self.step_optim(self.P[ii])
                self.step_optim(self.psi[ii])
                
                if self.ag_count[k] > 1:
                    
                    for l in range(self.ag_count[k] - 1):
                        
                        self.mu[ii + l+1] = copy.copy(self.mu[ii])
                        self.V_main[ii + l+1] = copy.copy(self.V_main[ii])
                        self.P[ii + l+1] = copy.copy(self.P[ii])
                        self.psi[ii + l+1] = copy.copy(self.psi[ii])
                
                ii += self.ag_count[k]
                
    def restrict_trade(self, mu):
        
        avg =  torch.mean( self.trade_coef * (torch.sum(mu[...,::2], 1) ** 2) )
                
        return avg
    
    def zero_trade(self, mu):
        
        mu[..., -2] = 0
        
        sums = -1 * torch.sum(mu[...,::2], 1)
        
        mu[..., -2] = sums
        
        return mu
        
    def update(self,n_iter=1, batch_size=256, epsilon=0.01, update='V', gen = True):
        
        t, S, X = self.__grab_mini_batch__(batch_size, epsilon)
        
        t = 0*t # start at time zero
        X = 0*X # start with 0 inventory ? 
     
        
        Y = self.__stack_state__(t, S, X)
        #pdb.set_trace()
        for i in range(self.env.N*self.env.T.size):
            
            
            #pdb.set_trace()
            # get actions
            MU = self.get_actions(Y, batch_size)
            
            # randomize actions -- separate randomizations on trade rates and probs
            
            MU = self.randomize_actions(MU, epsilon)
            
            if self.env.zero_sum:
                
                MU[...,(2*self.n_agents - 1)] = -1 * torch.sum(MU[...,0:2*(self.n_agents - 1):2], 1)
             
            
            #pdb.set_trace()
            
            Yp, r = self.env.step(Y, MU, self.epsilon, testing = False, gen = gen)
            
            mu, mu_p, V, Vp, P, P_p, psi, psi_p = self.get_value_advantage_mu(Y, Yp, batch_size)
            
            
            # ADD IN REPLAY BUFFER AND SAMPLING FROM IT
            
            MU_r = self.reorder_actions(MU)
            
            A = []
            Ap = []
            
            for k in range(self.n_agents):
                dmu = MU_r[k]-mu[k]
                A.append(- torch.einsum('...i,...ij,...j->...', dmu, P[k] , dmu) \
                         + torch.einsum('...i,...i->...', dmu[:,2:], psi[k]) )
                                    
                dmu_p = mu_p[k] - mu_p[k].detach()
                Ap.append(- torch.einsum('...i,...ij,...j->...', dmu_p, P_p[k] , dmu_p) \
                         + torch.einsum('...i,...i->...', dmu_p[:,2:], psi_p[k]) )
            
            # if i == self.env.N-4:
            #     pdb.set_trace()
            loss = 0
            not_done = 1 - 1*(i == (self.env.N-2))
            for k in range(self.n_agents):
                loss += torch.mean( (V[:,k] + A[k] - r[:,k]  \
                                     - not_done * self.gamma * (Vp[:,k].detach()) )**2 )  \
                                           + self.beta * torch.mean ( torch.abs(torch.sum(psi[k],1)) )
            
            trade_loss = self.trade_coef * self.restrict_trade(mu)
            
            mag = loss / (2 * trade_loss)
            pdb.set_trace()
            self.trade_coef = (1 - self.trade_soft) * self.trade_coef + self.trade_soft * mag
            
            loss += trade_loss
                
            self.zero_grad()
            
            loss.backward()
            
            self.VA_loss.append(loss.item())
            
            self.update_nets(update)
            
            Y = copy.copy(Yp.detach())
            
            # soft update main >> target
# =============================================================================
#             
#             for k in range(self.n_agents):
#                 self.soft_update(self.V_main[k]['net'], self.V_target[k]['net'])
#                 
# =============================================================================
            ii = 0
            for k in range(self.ag_count.size()[0]):
                self.soft_update(self.V_main[ii]['net'], self.V_target[ii]['net'])
                
                if self.ag_count[k] > 1:
                    
                    for l in range(self.ag_count[k] - 1):
                        self.V_main[ii + l+1] = copy.copy(self.V_main[ii])
                
                
                ii += k
                
                
            
    
    def update_random_time(self,n_iter=1, batch_size=256, epsilon=0.01, update='V', gen = True, it = 1):
        
        t, S, X = self.__grab_mini_batch__(batch_size, epsilon)
        
        Y = self.__stack_state__(t, S, X)
                
        # get actions
        #MU = self.mu['net'](Y)
        MU = self.get_actions(Y, batch_size)
            
        # randomize actions -- separate randomizations on trade rates and probs
        MU = self.randomize_actions(MU, epsilon)
        
        if self.env.zero_sum:
            
            #pdb.set_trace()
            
            MU = self.zero_trade(MU)
            
            #MU[...,(2*self.n_agents - 1)] = -1 * torch.sum(MU[...,0:2*(self.n_agents - 1):2], 1)
            
        
        Yp, r = self.env.step(Y, MU, flag = 1, epsilon = epsilon, testing = False, gen=gen, it = it)
            
        mu, mu_p, V, Vp, P, P_p, psi, psi_p = self.get_value_advantage_mu(Y, Yp, batch_size)
            
        # ADD IN REPLAY BUFFER AND SAMPLING FROM IT
            
        MU_r = self.reorder_actions(MU)
            
        A = []
        #Ap = []
            
        for k in range(self.n_agents):
            
            dmu = MU_r[k]-mu[k]
            A.append(- torch.einsum('...i,...ij,...j->...', dmu, P[k] , dmu) \
                     + torch.einsum('...i,...i->...', dmu[:,2:], psi[k]) )
                                    
            #dmu_p = mu_p[k] - mu_p[k].detach()
            #Ap.append(- torch.einsum('...i,...ij,...j->...', dmu_p, P_p[k] , dmu_p) \
            #          + torch.einsum('...i,...i->...', dmu_p[:,2:], psi_p[k]) )
            
            
        done = 1.0 * (torch.abs(Yp[:,0] - self.env.T[-1]) <= 1e-6)
        
        loss = 0
        trade_loss = 0
        
        for k in range(self.n_agents):
            #pdb.set_trace()
            loss += torch.mean( (V[:,k] + A[k] - r[:,k]  
                                - ( 1 - done) * self.gamma * (Vp[:,k].detach()) )**2  \
                                      + self.beta *  torch.sum(torch.abs(psi[k]), 1) )
                
        #only need to use one of the mu's in list  
        trade_loss += self.restrict_trade(mu[0])
        
        #pdb.set_trace()
        
        loss_full = loss + trade_loss
        
        
        mag = loss / (2 * trade_loss + 1)
        
        self.trade_coef = ((1 - self.trade_soft) * self.trade_coef + self.trade_soft * mag).detach()
        
        self.zero_grad()

        loss_full.backward()
            
        self.VA_loss.append(loss_full.item())
        
        self.update_nets(update)
            
        
        Y = copy.copy(Yp.detach())
            
        # soft update main >> target
        # =============================================================================
        #             
        #             for k in range(self.n_agents):
        #                 self.soft_update(self.V_main[k]['net'], self.V_target[k]['net'])
        #                 
        # =============================================================================
        
        ii = 0
        for k in range(self.ag_count.size()[0]):
            self.soft_update(self.V_main[ii]['net'], self.V_target[ii]['net'])
                        
            if self.ag_count[k] > 1:
                            
                for l in range(self.ag_count[k] - 1):
                    
                    #pdb.set_trace()
                    
                    self.V_main[ii + l+1] = copy.copy(self.V_main[ii])
                    
                        
            ii += self.ag_count[k]
            
    
    
    def train(self, n_iter=1_000, 
              batch_size=256, 
              n_plot=100,
              update_type = 'linear'):
        
        # self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
                
        C = 5000
        D = 10000
        
        if len(self.epsilon)==0:
            self.count=0
            
        gen_on = True
            
        for i in tqdm(range(n_iter)):
            
            
            epsilon = np.maximum(C/(D+len(self.epsilon)), 0.1)
            self.epsilon.append(epsilon)
            self.count += 1
            
# =============================================================================
#             if i == 30_000:
#                 gen_on = True
# =============================================================================
            
            if update_type == 'linear':

                #  self.update(batch_size=batch_size, epsilon=epsilon, update='all')
                self.update(batch_size=batch_size, epsilon=epsilon, update='V', gen = gen_on)
                self.update(batch_size=batch_size, epsilon=epsilon, update='A', gen = gen_on)
                self.update(batch_size=batch_size, epsilon=epsilon, update='mu', gen = gen_on)
                
            else:
                #self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='V')
                self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='all', gen = gen_on, it = i)
                #self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='A')
                #self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='mu')
            
            # if i == 1:
            #     pdb.set_trace()
            
            if np.mod(i+1,n_plot) == 0:
                
                self.loss_plots()
                self.run_strategy(1000, name= datetime.now().strftime("%H_%M_%S"), gen = gen_on)
                
        self.loss_plots()
        self.run_strategy(1000, name= datetime.now().strftime("%H_%M_%S"), gen = gen_on)
                
                #if self.n_agents == 1:
                #    self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
                
    def mv(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan
        
        for i in range(1,len(x)):
            
            if i < n+1:
                
                mu = np.mean(x[:i])
                mu2 = np.mean(x[:i]**2)
                
                y[i] = mu
                y_err[i] = np.sqrt(mu2-mu**2)
                
            else:
                
                mu = mu +(1.0/n) * (x[i-1]-x[i-n])
                mu2 = mu2 +(1.0/n) * (x[i-1]**2-x[i-n]**2)
                
                y[i] = mu
                y_err[i] = np.sqrt(mu2-mu**2)
                
        return y, y_err  
            
    def loss_plots(self):
        
        def plot(x, label, show_band=True):

            mv, mv_err = self.mv(np.array(x), 100)
        
            if show_band:
                plt.fill_between(np.arange(len(mv)), mv-mv_err, mv+mv_err, alpha=0.2)
            plt.plot(mv, label=label, linewidth=1) 
            plt.legend()
            plt.ylabel('loss')
            plt.yscale('symlog')
        
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,1,1)
        plot(self.VA_loss[50:], 'Loss', show_band=False)
        
        
        plt.tight_layout()
        # plt.show()
        
    
    def run_strategy(self, nsims=10_000, name="", N = None, gen = True):
        
        # N is time steps
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N * self.env.T.size + 1)).float().to(self.dev)
        X = torch.zeros((nsims, self.n_agents, N * self.env.T.size + 1)).float().to(self.dev)
        a = torch.zeros((nsims, (2 * self.n_agents), N * self.env.T.size)).float().to(self.dev)
        r = torch.zeros((nsims, self.n_agents, N * self.env.T.size)).float().to(self.dev)
        X_nu = torch.zeros((nsims, self.n_agents, N * self.env.T.size + 1)).float().to(self.dev)
        X_gen = torch.zeros((nsims, self.n_agents, N * self.env.T.size + 1)).float().to(self.dev)
        
        S[:,0] = self.env.S0
        X[:,:,0] = 0
        
        ones = torch.ones(nsims).to(self.dev)
        
        for k in range(N * self.env.T.size):
            
            Y = self.__stack_state__(self.env.t[k]* ones ,S[:,k], X[:,:,k])

            # normalize : Y (tSX)
            # get policy
            a[:,:,k] = self.get_actions(Y, nsims)
            
            if self.env.zero_sum:
                
                a[:,:,k] = self.zero_trade(a[:,:,k])
                
                #a[...,(2*self.n_agents - 1), k] = -1 * torch.sum(a[...,0:2*(self.n_agents - 1):2, k], 1)
             
            # step in environment
            
            #pdb.set_trace()
            
            Y_p, r[:,:,k], X_gen_t, X_nu_t = self.env.step(Y, a[:,:,k], flag = 0, epsilon = 0, testing = True, gen = gen, sim = True)
            
            # update subsequent state and inventory
            S[:, k+1] = Y_p[:,1]
            X[:,:, k+1] = Y_p[:,2:]
            
            # OCs traded, OCs generated for each agent at each step
            
            X_nu[:,:,k+1] = X_nu_t
            X_gen[:,:,k+1] = X_gen_t
            
        S = S.detach().cpu().numpy()
        X  = X.detach().cpu().numpy()
        X_nu  = X_nu.detach().cpu().numpy()
        X_gen  = X_gen.detach().cpu().numpy()

        a = a.detach().cpu().numpy()
        
        a = a.transpose(0,2,1)
        
        r = r.detach().cpu().numpy()

        plt.figure(figsize=(8,5))
        n_paths = 3
        
        # plots are broken... need to fix, just need to verify shapes of the tensors and network outputs
        #pdb.set_trace()
        
        
        def plot(t, x, plt_i, title , col = 'b' ):
            
            qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)

            plt.subplot(2, 3, plt_i)
            plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5, color = col)
            plt.plot(t, qtl[1,:], color=col, linewidth=1)
            #plt.plot(t, x[1,:], color=col, linewidth=1.5)
            
            #plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            plt.title(title)
            plt.xlabel(r"$t$")


        plot(self.env.t, (S), 1, r"$S_t$" )
        
        
        #colors = ['b','r','g','y','m','c', 'orange', 'skyblue', 'brown']
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:cyan', 'tab:grey']

        
        for ag in range(self.n_agents):
            
            plot(self.env.t, (X[:,ag,:]), 2, r"$X_t$", col = colors[ag])
            
            plot(self.env.t[:-1], np.cumsum(r[:,ag,:], axis=1), 3, r"$r_t$", col = colors[ag])
            
            # need to fix for shape of a
            plot(self.env.t[:-1], a[:,:,(2*ag)], 4, r"$\nu_t$", col = colors[ag])
            
            plot(self.env.t[:-1], a[:,:,(2*ag+1)], 5, r"$p_t$", col = colors[ag])

        plt.subplot(2, 3, 6)
        
        PnL = np.sum(r,axis=2)
        
        for ag in range(self.n_agents):
            
            pnl_sub = PnL[:,ag]
            
            if self.env.penalty == 'diff':
                #initial term
                naive_pen = torch.zeros(nsims)
                
                for l in range(self.env.T.size):
                    
                    if l == 0:
                        naive_pen += self.env.pen *self.env.R[ag] 
                    else:
                        step = (l) * self.env.N
                        #need to re-calculate the inv prior to compliance
                        X_comp = X[:,ag,(step-1)] + X_gen[:,ag,(step)] + X_nu[:,ag,(step)]
                        naive_pen += self.env.pen * torch.maximum(self.env.R[ag] - X_comp, torch.tensor(0))
                    
                pnl_ag = pnl_sub - naive_pen.numpy()
                #pnl_ag = pnl_sub
            else:
                pnl_ag = pnl_sub
                
            
            qtl = np.quantile(pnl_ag,[0.005, 0.5, 0.995])
            
            cvar = self.CVaR(pnl_ag, confidence_level = 0.95)
            #         
            plt.hist(pnl_ag, bins=np.linspace(qtl[0], qtl[-1], 51), density=True, color = colors[ag], alpha = 0.6)
            print("\n")
            print("Agent" , (ag+1) ,"Mean:" , qtl[1], ", Left Tail:", cvar)
            
        
        plt.tight_layout()
        plt.show()   
        
        
        plt.figure(figsize=(12 , 2 * self.n_agents))

        def plot_indiv(t, x, plt_i, title , clr = 'b' ):
            '''
            format
                 nu, p, x
            ag 1
            ag 2
            '''
            qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            plt.subplot(self.n_agents.item(), 3, plt_i)

            plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5, color = clr)
            plt.plot(t, qtl[1,:], color=clr, linewidth=1)
            #plt.plot(t, x[1,:], color=clr, linewidth=1.5)
            
            # n_paths = 3
            #plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            plt.title(title)

        
        for ag in range(self.n_agents):
            # inventory
            plot_indiv(self.env.t, (X[:,ag,:]), (3 + 3*ag), fr"$X_t$", clr = colors[ag])
                        
            # trade and prob
            plot_indiv(self.env.t[:-1], a[:,:,(2*ag)], (1 + 3*ag), r"$\nu_t$", clr = colors[ag])
            plot_indiv(self.env.t[:-1], a[:,:,(2*ag+1)], (2 + 3*ag), fr"$p_t$", clr = colors[ag])

        plt.figlegend(
            handles=[mpatches.Patch(color=colors[i], label=f'Agent {i+1}') for i in range(self.n_agents)], 
            loc='lower center',
            fancybox=True, shadow = True,
            ncol=self.n_agents
            )
        
        
        plt.tight_layout()
        plt.show()
        
        
        # return t, S, X, a, r
        return plt.gcf() #, performance
    
    
    
    
    
    
    
    def plot_nice(self, nsims=10_000, name="", N = None, gen = True):
        #make saved plots of individual items, eg: OC price, inventory, trade rates and probs
        
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N * self.env.T.size + 1)).float().to(self.dev)
        X = torch.zeros((nsims, self.n_agents, N * self.env.T.size + 1)).float().to(self.dev)
        a = torch.zeros((nsims, (2 * self.n_agents), N * self.env.T.size)).float().to(self.dev)
        r = torch.zeros((nsims, self.n_agents, N * self.env.T.size)).float().to(self.dev)
        X_nu = torch.zeros((nsims, self.n_agents, N * self.env.T.size + 1)).float().to(self.dev)
        X_gen = torch.zeros((nsims, self.n_agents, N * self.env.T.size + 1)).float().to(self.dev)
        
        
        S[:,0] = self.env.S0
        X[:,:,0] = 0
        
        ones = torch.ones(nsims).to(self.dev)
        
        for k in range(N * self.env.T.size):
            
            Y = self.__stack_state__(self.env.t[k]* ones ,S[:,k], X[:,:,k])

            # normalize : Y (tSX)
            # get policy
            
            a[:,:,k] = self.get_actions(Y, nsims)
            
            if self.env.zero_sum:
                
                a[:,:,k] = self.zero_trade(a[:,:,k])

            # step in environment
            
            pdb.set_trace()
            
            Y_p, r[:,:,k], X_gen_t, X_nu_t = self.env.step(Y, a[:,:,k], flag = 0, epsilon = 0, testing = True, gen = gen, sim = True)
            
            # update subsequent state and inventory
            S[:, k+1] = Y_p[:,1]
            X[:,:, k+1] = Y_p[:,2:]
            
            # OCs traded, OCs generated for each agent at each step
            
            X_nu[:,:,k+1] = X_nu_t
            X_gen[:,:,k+1] = X_gen_t
            
            
            
        S = S.detach().cpu().numpy()
        X  = X.detach().cpu().numpy()
        X_nu  = X_nu.detach().cpu().numpy()
        X_gen  = X_gen.detach().cpu().numpy()
        
        a = a.detach().cpu().numpy()
        
        a = a.transpose(0,2,1)
        
        r = r.detach().cpu().numpy()
        
        

        plt.figure()
        
        plt.plot()
        qtl= np.quantile(S, [0.01, 0.5, 0.99], axis=0)

        plt.fill_between(self.env.t, qtl[0,:], qtl[2,:], alpha=0.35, color = 'b')
        plt.plot(self.env.t, qtl[1,:], color='b', linewidth=1.5)
        for el in (self.env.T):
            plt.axvline(x=el, color = 'k', linestyle='dashed')
        plt.title(r'OC Price')
        plt.xlabel(r't')
        plt.ylabel(r'$S_t$')
        
        plt.savefig(str('oc_price.pdf'), format="pdf", bbox_inches="tight")
        
        def plot(t, dat, title, ylab, xlab, path, col, leg):
            
            plt.figure()
            
            #pdb.set_trace()
            
            for ag in range(self.n_agents):
                
                x = dat[:,ag,:]
                
            
                qtl= np.quantile(x, [0.01, 0.5, 0.99], axis=0)

                plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5, color = col[ag], label='_nolegend_')
                plt.plot(t, qtl[1,:], color = col[ag], linewidth=1.5)
                
            if leg:
                 plt.legend(['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5'],bbox_to_anchor=(1, 1))
            
            
            for el in (self.env.T):
                plt.axvline(x=el, color = 'k', linestyle='dashed')

            
            
            plt.title(title)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            
            
            
            plt.savefig(str(path), format="pdf", bbox_inches="tight")
            plt.show()

        def plot_action(t, dat, i, title, ylab, xlab, path, col, leg):
            
            # i = 0 or 1, if trade rate or prob, respectively
            
            plt.figure()
            
            shift = 0.000
            
            for ag in range(self.n_agents):
                
                x = dat[:,:,(2 * ag + i)]
                
            
                qtl= np.quantile(x, [0.01, 0.5, 0.99], axis=0)

                plt.fill_between(t, qtl[0,:] + ag * shift, qtl[2,:] + ag * shift, alpha=0.5, color = col[ag])
                plt.plot(t, qtl[1,:] + ag * shift, color=col[ag], linewidth=1.5)
            
            
                
                
            plt.title(title)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            
            
            if leg:
                plt.legend(['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'])
                
            for el in (self.env.T):
                plt.axvline(x=el, color = 'k', linestyle='dashed')
            
            plt.savefig(str(path), format="pdf", bbox_inches="tight")
            plt.show()
            
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:cyan']
        
        #pdb.set_trace()
        plot(self.env.t, X, "Agent Inventories", r"$X_t$", r"t", "inv.pdf", colors, True)
        
        plot_action(self.env.t[:-1], a, 0, "Agent Trade Rates", r"$\nu_t$", r"t", "trade.pdf", colors, False)
        plot_action(self.env.t[:-1], a, 1, "Agent Generation Probabilities", r"$p_t$", r"t", "prob.pdf", colors, False)
        
    
        
        PnL = np.sum(r,axis=2)
            
            
        fig, ax = plt.subplots(self.n_agents.item(), 3, figsize=(15, 15))
        
            
        for ag in range(self.n_agents):
            
            clr = colors[ag]
            
            
            qtl_nu = np.quantile(a[:,:,(2*ag)], [0.05, 0.5, 0.95], axis=0)
            qtl_p = np.quantile(a[:,:,(2*ag + 1)], [0.05, 0.5, 0.95], axis=0)
            qtl_inv = np.quantile(X[:,ag,:], [0.05, 0.5, 0.95], axis=0)
                
            # trade rate    
            ax[ag,0].plot(self.env.t[:-1], qtl_nu[1,:], color=clr, linewidth=1)
            ax[ag,0].fill_between(self.env.t[:-1], qtl_nu[0,:], qtl_nu[2,:], alpha=0.5, color = clr)
            ax[ag,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            for el in (self.env.T):
                ax[ag,0].axvline(x=el, color = 'k', linestyle='dashed')
            #prob
            ax[ag,1].plot(self.env.t[:-1], qtl_p[1,:], color=clr, linewidth=1)
            ax[ag,1].fill_between(self.env.t[:-1], qtl_p[0,:], qtl_p[2,:], alpha=0.5, color = clr)
            for el in (self.env.T):
                ax[ag,1].axvline(x=el, color = 'k', linestyle='dashed')
            ax[ag,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            #inventory
            ax[ag,2].plot(self.env.t, qtl_inv[1,:], color=clr, linewidth=1)
            ax[ag,2].fill_between(self.env.t, qtl_inv[0,:], qtl_inv[2,:], alpha=0.5, color = clr)
            for el in (self.env.T):
                ax[ag,2].axvline(x=el, color = 'k', linestyle='dashed')

            
            
            ax[ag,0].set(ylabel=r'Agent {}'.format(ag+1))
            
            
        ax[0,0].set_title(r'$\nu_t$')
        ax[0,1].set_title(r'$p_t$')
        ax[0,2].set_title(r'$X_t$')
            
           
            
        fig.align_ylabels(ax[:, 0])
        ax[-1, 1].set(xlabel=r't')
        
        plt.tight_layout()
        plt.savefig('solo_agents.pdf', format="pdf", bbox_inches="tight")
        plt.show()
        
        fig, ax = plt.subplots(1, self.n_agents.item(), figsize=(16, 8))

        
        for ag in range(self.n_agents):
            
            clr = colors[ag]
            
            pnl_sub = PnL[:,ag]
            
            if self.env.penalty == 'diff':
                #initial term
                naive_pen = torch.zeros(nsims)
                
                for l in range(self.env.T.size):
                    if l == 0:
                        naive_pen += self.env.pen *self.env.R[ag] 
                    else:
                        step = (l) * self.env.N
                        #need to re-calculate the inv prior to compliance
                        #as the inventory has already dropped
                        X_comp = X[:,ag,(step-1)] + X_gen[:,ag,(step)] + X_nu[:,ag,(step)]
                        naive_pen += self.env.pen * torch.maximum(self.env.R[ag] - X_comp, torch.tensor(0))
                    
                pnl_ag = pnl_sub - naive_pen.numpy()
                #pnl_ag = pnl_sub
            else:
                pnl_ag = pnl_sub
            
            qtl_pnl = np.quantile(pnl_ag,[0.001, 0.5, 0.999])
            cvar = self.CVaR(pnl_ag, confidence_level = 0.95)
            
            kde_pnl = stats.gaussian_kde(pnl_ag)
            xx = np.linspace(min(pnl_ag), max(pnl_ag), 100)
            
            #PnL
            ax[ag].hist(pnl_ag, bins=np.linspace(min(pnl_ag), max(pnl_ag), 20), density=True, color = clr, alpha = 0.5)
            ax[ag].plot(xx, kde_pnl(xx), color = clr)
            ax[ag].axvline(x=cvar, color = 'dimgrey', linestyle='dashed')
            
            
            ax[ag].set(xlabel=r'Agent {}'.format(ag+1))
            
# =============================================================================
#             ax[0,3].set_title(r'PnL')
# =============================================================================
            
            print("\n")
            print("Agent" , (ag+1) ,"True Mean:" , qtl_pnl[1], ", True Left Tail:", cvar )
            
            X_nu_ag = np.sum(X_nu[:,ag,:], axis = 1)
            X_gen_ag = np.sum(X_gen[:,ag,:], axis = 1)
            
            print("Agent" , (ag+1) ,"Total Trade:" , np.mean(X_nu_ag), ", Total Gen:", np.mean(X_gen_ag) )
            
            
           
        fig.suptitle('Agent PnL Histograms', fontsize = 16)
        
        plt.tight_layout()
        plt.savefig('pnl_hist.pdf', format="pdf", bbox_inches="tight")
        plt.show()
        


    