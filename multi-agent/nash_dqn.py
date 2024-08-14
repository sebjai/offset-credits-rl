# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from offset_env import offset_env as Environment

import numpy as np
import matplotlib.pyplot as plt

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
                 n_agents=2,
                 gamma=0.9999, beta = 100, alpha = 0, 
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
        
        self.__initialize_NNs__()
        
        self.t = []
        self.S = []
        self.X = []
        self.nu = []
        self.p = []
        self.r = []
        self.epsilon = []
        
        
        self.alpha = alpha
        
        self.VA_loss = []
        
# =============================================================================
#         self.Q_loss = []
#         self.pi_loss = []
# =============================================================================
        
    def reset(self, env):
        
        self.epsilon = []
        self.env = env
        self.lr = self.lr / 10
        
        #self.mu['net'].env = env
        #self.V_main['net'].env = env
        #self.V_target['net'].env = env
        
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
        
        for k in range(self.n_agents):
            self.V_main.append(create_net(n_in = (2 + self.n_agents), n_out = 1, n_nodes=32, n_layers=3))
            
            # try binary output activation instead of a probability.... might make graphs ugly but lets see
            self.mu.append(create_net(n_in = (2 + self.n_agents), n_out = 2, n_nodes=32, n_layers=3,
                                 out_activation=[lambda x : self.env.nu_max * torch.tanh(x),
                                                 lambda x : (torch.sigmoid(x))]))
            
            self.V_target.append(copy.copy(self.V_main[k]))
            
            self.psi.append(create_net(n_in=(2 + self.n_agents), n_out= (2 * (self.n_agents - 1)), n_nodes=32, n_layers=3))
            
            self.P.append(create_posdef_net(n_in=(2 + self.n_agents), n_agents=self.n_agents, n_nodes=32, n_layers=3))

        
        
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
            
            #pdb.set_trace()
            
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
            
            for k in range(self.n_agents):
                self.step_optim(self.mu[k])
                self.step_optim(self.V_main[k])
                self.step_optim(self.P[k])
                self.step_optim(self.psi[k])
                
    def restrict_trade(self, mu):
        
        sums = torch.sum(mu[...,::2], 1)
                
        return sums
        
    def update(self,n_iter=1, batch_size=256, epsilon=0.01, update='V'):
        
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
            
            #pdb.set_trace()
            
            Yp, r = self.env.step(Y, MU, self.epsilon, testing = False)
            
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
                
            self.zero_grad()
            
            loss.backward()
            
            self.VA_loss.append(loss.item())
            
            self.update_nets(update)
            
            Y = copy.copy(Yp.detach())
            
            # soft update main >> target
            
            for k in range(self.n_agents):
                self.soft_update(self.V_main[k]['net'], self.V_target[k]['net'])
            
    
    def update_random_time(self,n_iter=1, batch_size=256, epsilon=0.01, update='V'):
        
        t, S, X = self.__grab_mini_batch__(batch_size, epsilon)
        
        Y = self.__stack_state__(t, S, X)
                
        # get actions
        #MU = self.mu['net'](Y)
        MU = self.get_actions(Y, batch_size)
            
        # randomize actions -- separate randomizations on trade rates and probs
        MU = self.randomize_actions(MU, epsilon)

        Yp, r = self.env.step(Y, MU, flag = 1, epsilon = epsilon, testing = False)
            
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
        
        for k in range(self.n_agents):
            #pdb.set_trace()
            loss += torch.mean( (V[:,k] + A[k] - r[:,k]  
                                - ( 1 - done) * self.gamma * (Vp[:,k].detach()) )**2  \
                                      + self.beta *  torch.abs(torch.sum(psi[k],1)) )
        self.zero_grad()

        loss.backward()
            
        self.VA_loss.append(loss.item())
        
        self.update_nets(update)
            
        
        Y = copy.copy(Yp.detach())
            
        # soft update main >> target
            
        
        for k in range(self.n_agents):
            self.soft_update(self.V_main[k]['net'], self.V_target[k]['net'])
            
            
    
    
    def train(self, n_iter=1_000, 
              batch_size=256, 
              n_plot=100,
              update_type = 'linear'):
        
        # self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
        # self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
                
        C = 5000
        D = 10000
        
        if len(self.epsilon)==0:
            self.count=0
            
        for i in tqdm(range(n_iter)):
            
            
            epsilon = np.maximum(C/(D+len(self.epsilon)), 0.05)
            self.epsilon.append(epsilon)
            self.count += 1
            
            if update_type == 'linear':

                #  self.update(batch_size=batch_size, epsilon=epsilon, update='all')
                self.update(batch_size=batch_size, epsilon=epsilon, update='V')
                self.update(batch_size=batch_size, epsilon=epsilon, update='A')
                self.update(batch_size=batch_size, epsilon=epsilon, update='mu')
                
            else:
                #self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='V')
                self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='all')
                #self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='A')
                #self.update_random_time(batch_size=batch_size, epsilon=epsilon, update='mu')
            
            # if i == 1:
            #     pdb.set_trace()
            
            if np.mod(i+1,n_plot) == 0:
                
                self.loss_plots()
                self.run_strategy(1000, name= datetime.now().strftime("%H_%M_%S"))
                
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
        plot(self.VA_loss, 'Loss', show_band=False)
        
        
        plt.tight_layout()
        # plt.show()
        
    # TODO: Update run_strategy and the plots
    
    def run_strategy(self, nsims=10_000, name="", N = None):
        
        # N is time steps
        
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N * self.env.T.size + 1)).float().to(self.dev)
        X = torch.zeros((nsims, self.n_agents, N * self.env.T.size + 1)).float().to(self.dev)
        a = torch.zeros((nsims, (2 * self.n_agents), N * self.env.T.size)).float().to(self.dev)
        r = torch.zeros((nsims, self.n_agents, N * self.env.T.size)).float().to(self.dev)
        
        S[:,0] = self.env.S0
        X[:,:,0] = 0
        
        ones = torch.ones(nsims).to(self.dev)
        
        #pdb.set_trace()

        for k in range(N * self.env.T.size):
            
            
            
            Y = self.__stack_state__(self.env.t[k]* ones ,S[:,k], X[:,:,k])

            # normalize : Y (tSX)
            # get policy
            
            a[:,:,k] = self.get_actions(Y, nsims)

            # step in environment
            
            #pdb.set_trace()
            
            Y_p, r[:,:,k] = self.env.step(Y, a[:,:,k], flag = 0, epsilon = 0, testing = True)
            
           # pdb.set_trace()
            
            # update subsequent state and inventory
            S[:, k+1] = Y_p[:,1]
            X[:,:, k+1] = Y_p[:,2:]
            
        S = S.detach().cpu().numpy()
        X  = X.detach().cpu().numpy()

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
            
            #plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            plt.title(title)
            plt.xlabel(r"$t$")


        plot(self.env.t, (S), 1, r"$S_t$" )
        
        colors = ['b','r','g','y','m','c']
        
        for ag in range(self.n_agents):
            
            plot(self.env.t, (X[:,ag,:]), 2, r"$X_t$", col = colors[ag])
            plot(self.env.t[:-1], np.cumsum(r[:,ag,:], axis=1), 3, r"$r_t$", col = colors[ag])
            
            # need to fix for shape of a
            plot(self.env.t[:-1], a[:,:,(2*ag)], 4, r"$\nu_t$", col = colors[ag])
            plot(self.env.t[:-1], a[:,:,(2*ag+1)], 5, r"$p_t$", col = colors[ag])

        plt.subplot(2, 3, 6)
        
        PnL = np.sum(r,axis=2)
        #pdb.set_trace()
        
        #need to determine PnL shape for histograms of players...
        
# =============================================================================
#         if self.env.penalty in ('terminal', 'excess'):
#             
#             PnL = np.sum(r,axis=2)
#             
#         elif self.env.penalty =='diff':
#             naive_pen = self.env.pen * self.env.R * self.env.T.size
#             PnL = np.sum(r,axis=2) - naive_pen.numpy()
# =============================================================================
            
        for ag in range(self.n_agents):
            
            pnl_ag = PnL[:,ag]
            
            qtl = np.quantile(pnl_ag,[0.005, 0.5, 0.995])
            
            cvar = self.CVaR(pnl_ag, confidence_level = 0.95)
            #         
            plt.hist(pnl_ag, bins=np.linspace(qtl[0], qtl[-1], 51), density=True, color = colors[ag], alpha = 0.6)
            print("\n")
            print("Agent" , (ag+1) ,"Mean:" , qtl[1], ", Left Tail:", cvar)
            
        
        
        #plt.xlim(qtl[0], qtl[-1])
            #   
            
        plt.tight_layout()
            
        
        
       # plt.savefig("path_"  +self.name + "_" + name + ".pdf", format='pdf', bbox_inches='tight')
        plt.show()   
        
        t = 1.0* self.env.t
        
        # return t, S, X, a, r
        return plt.gcf() #, performance

    
    def plot_policy(self, name=""):
        '''
        plot policy for various states combinations at different time instances from 0 to self.env.T
        
        will work for single player or need to specifiy the player and update accordingly (eg: dims)

        '''
        
        NS = 75
        S = torch.linspace(self.env.S0-5*self.env.inv_vol,
                           self.env.S0+5*self.env.inv_vol, NS).to(self.dev)
        
        NX = 75
        X = torch.linspace(-1, self.env.X_max, NX).to(self.dev)
        
        Sm, Xm = torch.meshgrid(S, X,indexing='ij')
        Sm = Sm.to(self.dev)
        Xm = Xm.to(self.dev)

        def plot(k, lvls, title):
            
            # plot 
            t_steps = np.linspace(0, self.env.T[-1], 9)
            t_steps[-1] = (self.env.T[-1] - self.env.dt)
            
            n_cols = 3
            n_rows = int(np.floor(len(t_steps)/n_cols)+1)
            
            if n_rows*n_cols > len(t_steps):
                n_rows -= 1
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(6,5))
            
            plt.suptitle(title, y =1.01, fontsize = 'xx-large')
            
            for idx, ax in enumerate(axs.flat):
                
                
                t = torch.ones(NS,NX).to(self.dev) * t_steps[idx]
                Y = self.__stack_state__(t, Sm, Xm, plot1 = True)
                
                #temp_actions = self.get_actions(Y, batch_size)
                
                
                a = self.mu[0]['net'](Y).detach().squeeze().cpu().numpy()
                mask = (a[:,:,1]>0.999)
                a[mask,0] = np.nan
                cs = ax.contourf(Sm.cpu().numpy(), Xm.cpu().numpy(), a[:,:,k], 
                                  levels=lvls,
                                  cmap='RdBu')
                # print(torch.amin(a[:,:,0]),torch.amax(a[:,:,0]))
    
                ax.axvline(self.env.S0, linestyle='--', color='k')
                ax.axhline(self.env.R, linestyle='--', color='k')
                ax.axhline(0, linestyle='--', color='k')
                ax.set_title(r'$t={:.3f}'.format(t_steps[idx]) +'$',fontsize = 'x-large')
                ax.set_facecolor("gray")
            
            fig.text(0.5, -0.01, 'OC Price', ha='center',fontsize = 'x-large')
            fig.text(-0.01, 0.5, 'Inventory', va='center', rotation='vertical',fontsize = 'x-large')
            # fig.subplots_adjust(right=0.9)   
    
            cbar_ax = fig.add_axes([1.04, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(cs, cax=cbar_ax)
            # cbar.set_ticks(np.linspace(-self.env.nu_max/2, self.env.nu_max/2, 11))
            # cbar.set_ticks(np.linspace(-50, 50, 11))
                
            plt.tight_layout()
            
            plt.show()
            
            return fig
        
        trade_fig = plot(0, 
             np.linspace(-self.env.nu_max, self.env.nu_max, 21), 
             "Trade Rate Heatmap over Time")
        gen_fig = plot(1, 
             np.linspace(0,1,21),
             "Generation Probability Heatmap over Time")    
        return trade_fig, gen_fig



