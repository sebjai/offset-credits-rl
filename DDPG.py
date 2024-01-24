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

import copy

import pdb

from datetime import datetime

class pi_ann(nn.Module):

    def __init__(self, nNodes, nLayers, env=None):
        super(pi_ann, self).__init__()
        
        self.env = env
        
        self.nu = ann(3, 1, nNodes, nLayers, 
                      out_activation = [lambda x : self.env.nu_max*torch.tanh(x)],
                      env=env)
        
        self.prob = ann(3, 1, nNodes, nLayers,
                        out_activation = [torch.sigmoid],
                        env=env)

    def forward(self, Y):
        
        if len(Y.shape) == 2:
            y = torch.zeros(Y.shape[0],2)
        else:
            y = torch.zeros(Y.shape[0],Y.shape[1],2)
        
        y[...,0] = self.nu(Y).squeeze()
        y[...,1] = self.prob(Y).squeeze()
        
        return y
    
class DDPG():

    def __init__(self, env: Environment,  
                 gamma=0.9999,  
                 n_nodes=36, n_layers=3, 
                 lr=0.001, tau=0.001, sched_step_size = 100,
                 name=""):

        self.env = env
        self.gamma = gamma
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
        
        self.Q_loss = []
        self.pi_loss = []
        
    def reset(self, env):
        
        self.epsilon = []
        self.env = env
        self.pi['net'].env = env
        self.pi['net'].nu.env = env
        self.pi['net'].prob.env = env
        self.Q_main['net'].env = env
        self.Q_target['net'].env = env
        
        for g in self.pi['optimizer'].param_groups:
            g['lr'] = self.lr
        for g in self.Q_main['optimizer'].param_groups:
            g['lr'] = self.lr
        
        
    def __initialize_NNs__(self):
        
        # policy approximation
        #
        # features = t, S, X
        # out = nu, p
        #
        self.pi = {'net': pi_ann(nNodes=self.n_nodes, 
                                 nLayers=self.n_layers,
                                 env = self.env)}
        
        self.pi['optimizer'], self.pi['scheduler'] = self.__get_optim_sched__(self.pi)        
        
        # Q - function approximation
        #
        # features = t, S, X, nu, p
        # out = Q
        self.Q_main = {'net' : ann(n_in=5, 
                                  n_out=1,
                                  nNodes=self.n_nodes, 
                                  nLayers=self.n_layers,
                                  env = self.env) }

        self.Q_main['optimizer'], self.Q_main['scheduler'] = self.__get_optim_sched__(self.Q_main)
        
        self.Q_target = copy.deepcopy(self.Q_main)
        
        
    def __get_optim_sched__(self, net):
        
        optimizer = optim.AdamW(net['net'].parameters(),
                                lr=self.lr)
                    
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.sched_step_size,
                                              gamma=0.999)
    
        return optimizer, scheduler
    
    def soft_update(self, main, target):
    
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def __stack_state__(self, t, S, X):
        # normalization happens outside of stack state
        tS = torch.cat((t.unsqueeze(-1), 
                        S.unsqueeze(-1)), axis=-1)
        tSX = torch.cat((tS,
                         X.unsqueeze(-1)), axis=-1)
        return tSX
    
    
    def __grab_mini_batch__(self, batch_size, epsilon):
        t, S, X = self.env.randomize(batch_size, epsilon)
        return t, S, X
   
    def range_test(self, x, test='prob'):
        if test=='prob':
            if torch.amin(x) < 0 or torch.amax(x) > 1:
                print(torch.amin(x), torch.amax(x))

    def Update_Q(self, n_iter = 10, batch_size=256, epsilon=0.02, 
                 progress_bar=False):
        
        rg = range(n_iter) 
        if progress_bar:
            rg = tqdm(rg)
            
        for i in rg: 
            
            t, S, X = self.__grab_mini_batch__(batch_size, epsilon)
            
            
            # concatenate states
            Y = self.__stack_state__(t, S, X)
            
            # normalize : Y (tSX)
            # get pi (policy)
            a = self.pi['net'](Y).detach()

            # randomize actions
            a[:,0] += 0.5*self.env.nu_max*epsilon*torch.randn((batch_size,))
            a[:,0] = torch.clip(a[:,0], min=-self.env.nu_max, max=self.env.nu_max)
            a[:,1] += epsilon * torch.randn((batch_size,))
            a[:,1] = torch.clip(a[:,1], min=0, max=1)

            # get Q
            Q = self.Q_main['net'](Y, a )

            # step in the environment
            Y_p, r = self.env.step(Y, a)
            
            ind_T = 1.0 * (torch.abs(Y_p[:,0] - self.env.T) <= 1e-6).reshape(-1,1)

            # compute the Q(S', a*)
            # optimal policy at t+1
            a_p = self.pi['net'](Y_p).detach()
            
            # compute the target for Q
            Q_p = self.Q_target['net'](Y_p, a_p)
            target = r.reshape(-1,1) + (1.0 - ind_T) * self.gamma * Q_p

            loss = torch.mean((target.detach() - Q)**2)
            
            # compute the gradients
            self.Q_main['optimizer'].zero_grad()
            
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.Q_main['net'].parameters(), 1)

            # perform step using those gradients
            self.Q_main['optimizer'].step()                
            
            
            self.Q_loss.append(loss.item())
            
            self.soft_update(self.Q_main['net'], self.Q_target['net'])
        
        self.Q_main['scheduler'].step() 
        # self.Q_target = copy.deepcopy(self.Q_main)
        
    def Update_pi(self, n_iter = 10, batch_size=256, epsilon=0.02):

        for i in range(n_iter):
            
            t, S, X = self.__grab_mini_batch__(batch_size, epsilon)
            
            # concatenate states 
            Y = self.__stack_state__(t, S, X)

            a = self.pi['net'](Y)
            
            Q = self.Q_main['net'](Y,a )
            
            loss = -torch.mean(Q)
                
            self.pi['optimizer'].zero_grad()
            
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.pi['net'].parameters(), 1)
            
            self.pi['optimizer'].step()
            
            self.pi_loss.append(loss.item())
            
        self.pi['scheduler'].step()
            
    def Update_Q_pi(self, n_iter = 10, batch_size=256, epsilon=0.02):
        
        for i in range(n_iter): 
            
            t, S, X = self.__grab_mini_batch__(batch_size, epsilon)
            
            t *= 0
            S[:] = self.env.S0
            X[:] = 0
            
            # concatenate states
            Y = self.__stack_state__(t, S, X)
            
            for j in range(self.env.N-1):
            
                # normalize : Y (tSX)
                # get pi (policy)
                a = self.pi['net'](Y)
                a_cp = a.clone()
                a = a.detach()
    
                # randomize actions
                a[:,0] += 10*epsilon*torch.randn((batch_size,))
                a[:,0] = torch.clip(a[:,0], min=-self.env.nu_max, max=self.env.nu_max)
                
                a[:,1] += 0.5*epsilon * torch.randn((batch_size,))
                a[:,1] = torch.clip(a[:,1], min=0, max=1)
    
                # get Q
                Q = self.Q_main['net'](Y, a )
    
                # step in the environment
                Y_p, r = self.env.step(Y, a)
                
                ind_T = 1.0 * (torch.abs(Y_p[:,0] - self.env.T) <= 1e-6).reshape(-1,1)
    
                # compute the Q(S', a*)
                # optimal policy at t+1
                a_p = self.pi['net'](Y_p).detach()
                
                # compute the target for Q
                Q_p = self.Q_target['net'](Y_p, a_p)
                target = r.reshape(-1,1) + (1.0 - ind_T) * self.gamma * Q_p
    
                loss = torch.mean((target.detach() - Q)**2)
                
                # compute the gradients
                self.Q_main['optimizer'].zero_grad()
                
                loss.backward()
                
                # torch.nn.utils.clip_grad_norm_(self.Q_main['net'].parameters(), 1)
    
                # perform step using those gradients
                self.Q_main['optimizer'].step()                
                self.Q_main['scheduler'].step() 
                
                self.Q_loss.append(loss.item())
                
                # update pi 
                Q = self.Q_main['net'](Y, a_cp )
                
                loss = -torch.mean(Q)
                    
                self.pi['optimizer'].zero_grad()
                
                loss.backward()
                
                # torch.nn.utils.clip_grad_norm_(self.pi['net'].parameters(), 1)
                
                self.pi['optimizer'].step()
                self.pi['scheduler'].step()
                
                self.pi_loss.append(loss.item())
                
                Y = Y_p.detach().clone()
                
                self.soft_update(self.Q_main['net'], self.Q_target['net'])   
            
            
    def train(self, n_iter=1_000, 
              n_iter_Q=10, 
              n_iter_pi=5, 
              batch_size=256, 
              n_plot=100):
        
        self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
        self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))

        C = 100
        D = 200
        
        if len(self.epsilon)==0:
            self.count=0
            
        print("burning in the Q function for the initial policy...")
        self.Update_Q(n_iter=1000, 
                      batch_size=batch_size, 
                      epsilon=0.5, progress_bar=True)
        
        print("now performing full updates Q > pi...")
        
        for i in tqdm(range(n_iter)):
            
            epsilon = np.maximum(C/(D+len(self.epsilon)), 0.02)
            self.epsilon.append(epsilon)
            self.count += 1

            self.Update_Q(n_iter=n_iter_Q, 
                          batch_size=batch_size, 
                          epsilon=epsilon)
            # pdb.set_trace()
            self.Update_pi(n_iter=n_iter_pi, 
                            batch_size=batch_size, 
                            epsilon=epsilon)
            
            # self.Update_Q_pi(n_iter=1, 
            #                   batch_size=batch_size, 
            #                   epsilon=epsilon)

            if np.mod(i+1,n_plot) == 0:
                
                self.loss_plots()
                self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))
                self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
                
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
        plt.subplot(1,2,1)
        plot(self.Q_loss, r'$Q$', show_band=False)
        
        plt.subplot(1,2,2)
        plot(self.pi_loss, r'$\pi$')
        
        plt.tight_layout()
        plt.show()
        
    def run_strategy(self, nsims=10_000, name="", N = None):
        
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N)).float()
        X = torch.zeros((nsims, N)).float()
        a = torch.zeros((nsims, 2, N-1)).float()
        r = torch.zeros((nsims, N-1)).float()


        S[:,0] = self.env.S0
        X[:,0] = 0
        
        ones = torch.ones(nsims)

        for k in range(N-1):
            
            Y = self.__stack_state__(self.env.t[k]* ones ,S[:,k], X[:,k])

            # normalize : Y (tSX)
            # get policy
            a[:,:,k] = self.pi['net'](Y)

            # step in environment
            Y_p, r[:,k] = self.env.step(Y, a[:,:,k], flag = 0)
        
            # update subsequent state and inventory
            S[:, k+1] = Y_p[:,1]
            X[:, k+1] = Y_p[:,2]
            
        S = S.detach().numpy()
        X  = X.detach().numpy()
        
        a = a.detach().numpy()
        
        a = a.transpose(0,2,1)
        mask = (a[:,:,1] > 0.5)
        a[mask,0] = 0
        
        r = r.detach().numpy()

        plt.figure(figsize=(8,5))
        n_paths = 3
        
        def plot(t, x, plt_i, title ):
            
            qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)

            plt.subplot(2, 3, plt_i)
            plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5)
            plt.plot(t, qtl[1,:], color='k', linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            plt.title(title)
            plt.xlabel(r"$t$")


        plot(self.env.t, (S), 1, r"$S_t$" )
        plot(self.env.t, X, 2, r"$X_t$")
        plot(self.env.t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")
        plot(self.env.t[:-1], a[:,:,0], 4, r"$\nu_t$")
        plot(self.env.t[:-1], a[:,:,1], 5, r"$p_t$")

        plt.subplot(2, 3, 6)
        
        naive_pen = self.env.pen*self.env.R
        
        if self.env.penalty =='terminal':
            PnL = np.sum(r,axis=1)
        elif self.env.penalty =='diff':
            PnL = np.sum(r,axis=1) - naive_pen
            
        qtl = np.quantile(PnL,[0.005, 0.5, 0.995])
        
        plt.hist(PnL, bins=np.linspace(qtl[0], qtl[-1], 101), density=True)
        plt.axvline(qtl[1], color='g', linestyle='--', linewidth=2)
        plt.axvline(-naive_pen, color='r', linestyle='--', linewidth=2)
        plt.xlim(qtl[0], qtl[-1])
        
        plt.tight_layout()
        
       # plt.savefig("path_"  +self.name + "_" + name + ".pdf", format='pdf', bbox_inches='tight')
        plt.show()   
        
        t = 1.0* self.env.t
        
        return t, S, X, a, r

    def plot_policy(self, name=""):
        '''
        plot policy for various states combinations at different time instances from 0 to self.env.T

        '''
        
        NS = 51
        S = torch.linspace(self.env.S0-3*self.env.inv_vol,
                           self.env.S0+3*self.env.inv_vol, NS)
        
        NX = 51
        X = torch.linspace(-1, self.env.X_max, NX)
        
        Sm, Xm = torch.meshgrid(S, X,indexing='ij')

        def plot(k, lvls, title):
            
            # plot 
            t_steps = np.linspace(0, self.env.T, 9)
            t_steps[-1] = (self.env.T - self.env.dt)
            
            n_cols = 3
            n_rows = int(np.floor(len(t_steps)/n_cols)+1)
            
            if n_rows*n_cols > len(t_steps):
                n_rows -= 1
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(6,5))
            
            plt.suptitle(title, y =1.01, fontsize = 'xx-large')
            
            for idx, ax in enumerate(axs.flat):
                t = torch.ones(NS,NX) * t_steps[idx]
                Y = self.__stack_state__(t, Sm, Xm)
                
                # normalize : Y (tSX)
                a = self.pi['net'](Y).detach().squeeze().numpy()
                mask = (a[:,:,1]>0.999)
                a[mask,0] = np.nan
                cs = ax.contourf(Sm.numpy(), Xm.numpy(), a[:,:,k], 
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
        
        plot(0, 
             np.linspace(-self.env.nu_max, self.env.nu_max, 21), 
             "Trade Rate Heatmap over Time")
        plot(1, 
             np.linspace(0,1,21),
             "Generation Probability Heatmap over Time")    
