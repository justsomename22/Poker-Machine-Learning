# replay_buffer.py
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Ensure we don't try to sample more than what's available
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.array(actions), np.array(rewards),
                np.stack(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)