import numpy as np
import gymnasium as gym
import time 

env = gym.make('CartPole-v1', render_mode='human')

state, info = env.reset()
done = False

while not done:
    # 渲染画面
    env.render()
    
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 与环境交互
    next_state, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    state = next_state
    
    # 稍微暂停一下，否则画面运行太快，一闪而过
    time.sleep(0.05) 

env.close()
