import gymnasium as gym
import torch
from dqn_agent import Agent

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="../videos", episode_trigger=lambda x: x < 5)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = Agent(state_size=state_size, action_size=action_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('../checkpoints/checkpoint_best.pth'))

print("Recording 5 episodes...")

for i in range(5):
    state, info = env.reset()
    done = False
    score = 0
    
    while not done:
        action = agent.act(state, eps=0.0)
        state, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        
    print(f"Episode {i+1} finished with score: {score:.2f}")

env.close()
print("Finished! Check the 'videos' folder for your MP4 files.")