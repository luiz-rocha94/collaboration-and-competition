from ddpg_agent import Agent, ReplayBuffer
import numpy as np

import torch


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 15         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

class PADDPG:
    def __init__(self, state_size, action_size, n_agents, random_seed):
        super(PADDPG, self).__init__()
        self.paddpg_agent = [Agent(state_size, action_size, state_size+action_size, random_seed, f'a{i}') for i in range(n_agents)]
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def act(self, states):
        last_states = states[:,-8:]
        actions = np.empty((0,2))
        for agent, last_state in zip(self.paddpg_agent, last_states):
            action = agent.act(last_state)
            actions = np.vstack((actions, action))
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        last_states = states[:,-8:]
        last_next_states = next_states[:,-8:]
        # Save experience / reward
        for state, action, reward, next_state, done in zip(last_states, actions, rewards, last_next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
            
    def reset(self):
        for agent in self.paddpg_agent:
            agent.reset()
            
    def learn(self):
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for agent in self.paddpg_agent:
                experiences = self.memory.sample()
                agent.learn(experiences, GAMMA)
            
    def save(self):
        for agent in self.paddpg_agent:
            torch.save(agent.actor_local.state_dict(), f'{agent.name}_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'{agent.name}_checkpoint_critic.pth')
            
    def load(self):
        for agent in self.paddpg_agent:
            agent.actor_local.load_state_dict(torch.load(f'{agent.name}_checkpoint_actor.pth'))
            agent.critic_local.load_state_dict(torch.load(f'{agent.name}_checkpoint_critic.pth'))  
            