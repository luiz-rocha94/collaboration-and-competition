# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:44:58 2021

@author: rocha
"""

from ddpg_agent import Agent, ReplayBuffer
import numpy as np

import torch
import torch.nn.functional as F


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 30         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

class MADDPG:
    def __init__(self, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()
        self.maddpg_agent = [Agent(state_size*2, action_size, random_seed, 'a1'), 
                             Agent(state_size*2, action_size, random_seed, 'a2'),]
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def act(self, states):
        flatten_state = states[:,-8:].flatten()
        actions = np.empty((0,2))
        for agent in self.maddpg_agent:
            action = agent.act(flatten_state)
            actions = np.vstack((actions, action))
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        flatten_state = states[:,-8:].flatten()
        flatten_action = actions.flatten()
        flatten_next_state = next_states[:,-8:].flatten()
        # Save experience / reward
        self.memory.add(flatten_state, flatten_action, rewards, flatten_next_state, dones)   
            
    def reset(self):
        for agent in self.maddpg_agent:
            agent.noise.reset()
            
    def local_act(self, states):
        """get actions from all agents in the MADDPG object"""
        local_actions = [torch.cat([agent.actor_local(state) for agent in self.maddpg_agent], dim=0).unsqueeze(0) for state in states]
        return torch.cat(local_actions, dim=0)    

    def target_act(self, next_states):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [torch.cat([agent.actor_target(next_state) for agent in self.maddpg_agent], dim=0).unsqueeze(0) for next_state in next_states]
        return torch.cat(target_actions, dim=0)    
            
    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for i, agent in enumerate(self.maddpg_agent):
                experiences = self.memory.sample()
                states, actions, rewards, next_states, dones = experiences
                rewards = rewards[:,i].unsqueeze(1)
                dones = dones[:,i].unsqueeze(1)
        
                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models
                actions_next = self.target_act(next_states)
                Q_targets_next = agent.critic_target(next_states, actions_next)
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
                # Compute critic loss
                Q_expected = agent.critic_local(states, actions)
                critic_loss = F.mse_loss(Q_expected, Q_targets)
                # Minimize the loss
                agent.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                agent.critic_optimizer.step()
        
                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actions_pred = self.local_act(states)
                actor_loss = -agent.critic_local(states, actions_pred).mean()
                # Minimize the loss
                agent.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                agent.actor_optimizer.step()
            
            for agent in self.maddpg_agent:
                # ----------------------- update target networks ----------------------- #
                self.soft_update(agent.critic_local, agent.critic_target, TAU)
                self.soft_update(agent.actor_local, agent.actor_target, TAU)     
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)        
            
    def save(self):
        for agent in self.maddpg_agent:
            torch.save(agent.actor_local.state_dict(), f'{agent.name}_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'{agent.name}_checkpoint_critic.pth')
            
    def load(self):
        for agent in self.maddpg_agent:
            agent.actor_local.load_state_dict(torch.load(f'{agent.name}_checkpoint_actor.pth'))
            agent.critic_local.load_state_dict(torch.load(f'{agent.name}_checkpoint_critic.pth'))  
            