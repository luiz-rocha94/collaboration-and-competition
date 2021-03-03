from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from maddpg import MADDPG

file_name   = r'D:\deep-reinforcement-learning\p3_collab-compet\Tennis_Windows_x86_64\Tennis.exe'
env         = UnityEnvironment(file_name=file_name)  # open environment
brain_name  = env.brain_names[0]                     # get the default brain
brain       = env.brains[brain_name]
env_info    = env.reset(train_mode=True)[brain_name] # reset the environment
num_agents  = len(env_info.agents)                   # number of agents
action_size = brain.vector_action_space_size         # size of each action
states      = env_info.vector_observations           # examine the state space 
state_size  = states.shape[1]
# create the agent
agents = MADDPG(state_size=state_size, action_size=action_size, n_agents=num_agents, random_seed=4)

def maddpg(n_episodes=3000):
    scores_deque      = deque(maxlen=100) # last 100 scores
    scores            = []                # all scores  
    average_scores    = []                # all average scores         
    max_average_score = 0                 # max average score
    for i_episode in range(1, n_episodes+1):
        agents.reset()                                           # reset noise    
        env_info       = env.reset(train_mode=True)[brain_name]  # reset the environment    
        states         = env_info.vector_observations            # get the current state
        episode_scores = np.zeros(num_agents)                    # initialize the score
        while True:
            actions     = agents.act(states)                    # select an action
            env_info    = env.step(actions)[brain_name]         # send action to tne environment
            next_states = env_info.vector_observations          # get next state
            rewards     = env_info.rewards                      # get reward
            dones       = env_info.local_done                   # see if episode finished
            agents.step(states, actions, rewards, next_states,
                        dones)                                  # Save experience and learn
            episode_scores += rewards                           # update the score
            states          = next_states                       # roll over state to next time step
            if np.any(dones):                                   # exit loop if episode finished
                break
        score = np.max(episode_scores)                          # max episode score
        scores_deque.append(score)      
        scores.append(score)
        average_score = np.mean(scores_deque)                   # average score
        average_scores.append(average_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, average_score, score), end="")
        if average_score > max_average_score and average_score >= 0.5:
            # Save best agent
            agents.save()
        max_average_score = max(max_average_score, average_score)
    return scores, average_scores

scores, average_scores = maddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.plot(np.arange(1, len(scores)+1), average_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

"""
agents.load()
agents.reset()                                           # reset noise    
env_info       = env.reset(train_mode=False)[brain_name] # reset the environment    
states         = env_info.vector_observations            # get the current state
episode_scores = np.zeros(num_agents)                    # initialize the score
while True:
    actions         = agents.act(states)                 # select an action
    env_info        = env.step(actions)[brain_name]      # send action to tne environment
    next_states     = env_info.vector_observations       # get next state
    rewards         = env_info.rewards                   # get reward
    dones           = env_info.local_done                # see if episode finished
    episode_scores += rewards                            # update the score
    states          = next_states                        # roll over state to next time step
    score = np.max(episode_scores)                       # max episode score
    print('\rScore: {:.2f}'.format(score), end="")
    if np.any(dones):                                    # exit loop if episode finished
        break
"""
env.close()
