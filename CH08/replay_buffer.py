from collections import deque
import random
import numpy as np
import gymnasium as gym
import time 

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done

env = gym.make('CartPole-v1', render_mode='human')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state, info = env.reset()
    done = False

    while not done:
        env.render()

        action = 0        
        next_state, reward, terminated, truncated, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        done = terminated or truncated
        
        time.sleep(0.05) 

env.close()

state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.get_batch()
print(state_batch.shape)      # (32, 4)
print(action_batch.shape)     # (32,)
print(reward_batch.shape)     # (32,)
print(next_state_batch.shape) # (32, 4)
print(done_batch.shape)       # (32,)
