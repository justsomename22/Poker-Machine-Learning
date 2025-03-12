# replay_buffer.py
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store None as a placeholder for terminal states
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Ensure we don't try to sample more than what's available
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert states to numpy array
        states = np.stack(states)
        
        # Handle next_states - replace None with zeros for terminal states
        valid_next_states = []
        for next_state in next_states:
            if next_state is not None:
                valid_next_states.append(next_state)
            else:
                # Create a zero array with the same shape as states
                valid_next_states.append(np.zeros_like(states[0]))
        
        return (states, np.array(actions), np.array(rewards),
                np.stack(valid_next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)