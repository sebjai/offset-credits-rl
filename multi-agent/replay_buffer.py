#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:46:52 2024

@author: liamwelsh
"""

import torch

class replay_buffer():

    def __init__(self, directory: str, max_size = 10_000, input_shape = 9, save_frequency = 50, reset_buffer = False):

        self.mem_size = int(max_size)
        self.file_path = directory
        self.save_frequency  = save_frequency

        if reset_buffer:
            
            self.mem_cntr = 0
            self.state_memory = torch.zeros(size = (self.mem_size, input_shape))
            self.new_state_memory = torch.zeros(size = (self.mem_size, input_shape))
            self.action_memory = torch.zeros(size = (self.mem_size, 2)) #should be 2 * n_agents
            self.reward_memory = torch.zeros(size = (self.mem_size, 1)) # should be 1 * n_agents
            self.terminal_memory = torch.zeros(size = (self.mem_size, 1), dtype=torch.bool)
            
            torch.save(torch.tensor(self.mem_cntr), self.file_path + '/mem_cntr.pt')
            torch.save(self.state_memory.detach(), self.file_path + '/state.pt')
            torch.save(self.new_state_memory.detach(), self.file_path + '/state_.pt')
            torch.save(self.action_memory.detach(), self.file_path + '/action.pt')
            torch.save(self.reward_memory.detach(), self.file_path + '/reward.pt')
            torch.save(self.terminal_memory.detach(), self.file_path + '/terminal.pt')

        else:
            self.mem_cntr = int(torch.load(self.file_path + '/mem_cntr.pt'))   
            self.state_memory = torch.load(self.file_path + '/state.pt')
            self.new_state_memory = torch.load(self.file_path + '/state_.pt')
            self.action_memory = torch.load(self.file_path + '/action.pt')
            self.reward_memory = torch.load(self.file_path + '/reward.pt')
            self.terminal_memory = torch.load(self.file_path + '/terminal.pt')

        self.mem_cntr_initial = self.mem_cntr
        
    #Consider saving transitions to disk torch.save

    def store_transition(self, state, action, reward, new_state, done):
        '''Store a state, aciton, reward, and new state to the buffer
        '''

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.detach()
        self.new_state_memory[index] = new_state.detach()
        self.action_memory[index] = action.detach()
        self.reward_memory[index] = reward.detach()
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        '''Randomly sample from the buffer
        '''
        
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = torch.ones(max_mem).multinomial(num_samples = batch_size, replacement = False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
    def save_buffer(self):
        '''Save the current buffer onto the disk
        '''

        torch.save(torch.tensor(self.mem_cntr), self.file_path + '/mem_cntr.pt')
        torch.save(self.state_memory.detach(), self.file_path + '/state.pt')
        torch.save(self.new_state_memory.detach(), self.file_path + '/state_.pt')
        torch.save(self.action_memory.detach(), self.file_path + '/action.pt')
        torch.save(self.reward_memory.detach(), self.file_path + '/reward.pt')
        torch.save(self.terminal_memory.detach(), self.file_path + '/terminal.pt')
        