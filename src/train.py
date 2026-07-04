import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

from dqn_agent import Agent

env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    """Deep Q-Learning Training Loop."""
    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start                    
    
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break 
        
        scores_window.append(score)       
        scores.append(score)              
        eps = max(eps_end, eps_decay * eps) 
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        if np.mean(scores_window) >= 250.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '../checkpoints/checkpoint_best.pth')
            break
            
    return scores

if __name__ == "__main__":
    print("Starting training! This may take a while depending on your CPU/GPU...")
    
    scores = dqn()
    
    env.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Lunar Lander Training Progress')
    plt.show()