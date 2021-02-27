from ddpg_agent import Agent, BATCH_SIZE, GAMMA
import numpy as np

import torch


class PADDPG:
    def __init__(self, state_size, action_size, n_agents, random_seed):
        super(PADDPG, self).__init__()
        self.paddpg_agent = [Agent(state_size, action_size, state_size+action_size, random_seed, f'a{i}') for i in range(n_agents)]
        
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
        for agent, state, action, reward, next_state, done in zip(self.paddpg_agent, last_states, actions, rewards, last_next_states, dones):
            agent.step(state, action, reward, next_state, done)
            
    def reset(self):
        for agent in self.paddpg_agent:
            agent.reset()
            
    def learn(self):
        for agent in self.paddpg_agent:
            # Learn, if enough samples are available in memory
            if len(agent.memory) > BATCH_SIZE:
                experiences = agent.memory.sample()
                agent.learn(experiences, GAMMA)
            
    def save(self):
        for agent in self.paddpg_agent:
            torch.save(agent.actor_local.state_dict(), f'{agent.name}_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'{agent.name}_checkpoint_critic.pth')
            
    def load(self):
        for agent in self.paddpg_agent:
            agent.actor_local.load_state_dict(torch.load(f'{agent.name}_checkpoint_actor.pth'))
            agent.critic_local.load_state_dict(torch.load(f'{agent.name}_checkpoint_critic.pth'))  
            