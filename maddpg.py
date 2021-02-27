from ddpg_agent import Agent, ReplayBuffer, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU   
import numpy as np

import torch
import torch.nn.functional as F


class MADDPG:
    def __init__(self, state_size, action_size, n_agents, random_seed):
        super(MADDPG, self).__init__()
        self.maddpg_agent = [Agent(state_size, action_size, action_size*n_agents, 
                                   random_seed, f'a{i}') for i in range(n_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def act(self, states):
        last_state = states[:,-8:]
        actions = np.empty((0,2))
        for agent, state in zip(self.maddpg_agent, last_state):
            action = agent.act(state)
            actions = np.vstack((actions, action))
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        last_states = states[:, -8:].flatten()
        actions = actions.flatten()
        last_next_states = next_states[:, -8:].flatten()
        # Save experience / reward
        self.memory.add(last_states, actions, rewards, last_next_states, dones)   
            
    def reset(self):
        for agent in self.maddpg_agent:
            agent.noise.reset()
            
    def local_act(self, agent_i, states):
        """get actions from all agents in the MADDPG object"""
        target_actions = [agent.actor_local(state) if i == agent_i else agent.actor_local(state).detach() 
                          for i, agent, state in zip([0, 1], self.maddpg_agent, states)]
        return torch.cat(target_actions, dim=-1)

    def target_act(self, next_states):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.actor_target(next_state) for agent, next_state in zip(self.maddpg_agent, next_states)]
        return torch.cat(target_actions, dim=-1)
            
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
                states = [states[:, :8], states[:, 8:]]
                rewards = rewards[:,i].unsqueeze(1)
                next_states = [next_states[:, :8], next_states[:, 8:]]
                dones = dones[:,i].unsqueeze(1)
        
                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models
                actions_next = self.target_act(next_states)
                Q_targets_next = agent.critic_target(next_states[i], actions_next)
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
                # Compute critic loss
                Q_expected = agent.critic_local(states[i], actions)
                critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
                # Minimize the loss
                agent.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                agent.critic_optimizer.step()
        
                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actions_pred = self.local_act(i, states)
                actor_loss = -agent.critic_local(states[i], actions_pred).mean()
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
            