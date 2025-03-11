# self_play.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from environment import PokerEnv
from model import init_model
from replay_buffer import ReplayBuffer
from utils import log_episode
import torch.backends.cudnn as cudnn
import os
from torch.cuda.amp import autocast, GradScaler

class AIPlayer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def act(self, obs, legal_actions, epsilon=0.0):
        """Choose an action based on observation and legal actions"""
        if np.random.random() < epsilon:
            return np.random.choice(legal_actions)
        
        with torch.no_grad():
            self.model.eval()
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).squeeze(0)
            legal_q_values = q_values[legal_actions]
            action = legal_actions[torch.argmax(legal_q_values).item()]
            self.model.train()
            return action

def load_model_with_compatibility(model_path, input_size, output_size, device):
    """Load a model with compatibility handling for architecture changes"""
    model = init_model(input_size, output_size, device)
    
    # Load the state dict
    state_dict = torch.load(model_path)
    
    # Filter out unexpected keys
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    # Update the model dict with filtered state dict
    model_dict.update(filtered_state_dict)
    
    # Load the filtered state dict
    model.load_state_dict(model_dict, strict=False)
    return model

def train_with_self_play(episodes=222001, batch_size=1024, gamma=0.97, epsilon_start=0.3, 
                         epsilon_end=0.1, epsilon_decay=0.9995, start_episode=106000,
                         pretrained_model_path="model_episode_106000.pth"):
    # Set up GPU for optimal performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        cudnn.benchmark = True  # Optimize for fixed input sizes (faster)
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        
        # Optional: Set environment variables for better GPU performance
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # More helpful error messages
    else:
        print("Training on CPU")
    
    # Initialize environment
    env = PokerEnv()
    
    # Initialize main agent model from pretrained weights
    print(f"Loading pretrained model from {pretrained_model_path}")
    main_model = load_model_with_compatibility(pretrained_model_path, 
                                              input_size=54, 
                                              output_size=env.num_actions, 
                                              device=device)
    
    # Initialize target model with the same weights
    target_model = init_model(input_size=54, output_size=env.num_actions, device=device)
    target_model.load_state_dict(main_model.state_dict())
    
    # Initialize opponent model with the same weights
    opponent_model = init_model(input_size=54, output_size=env.num_actions, device=device)
    opponent_model.load_state_dict(main_model.state_dict())
    
    # Create AI players
    main_agent = AIPlayer(main_model, device)
    opponent_agent = AIPlayer(opponent_model, device)
    
    # Setup optimizer
    optimizer = optim.Adam(main_model.parameters(), lr=0.00005)  # Lower learning rate for fine-tuning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    replay_buffer = ReplayBuffer(capacity=100000)  # Larger buffer for continued training

    # Start with a lower epsilon since we're continuing training
    epsilon = epsilon_start
    opponent_epsilon = 0.2  # Increased from 0.1
    
    # Parameters for randomized opponent updates
    base_update_freq = 100  # Base frequency for updates
    update_prob = 0.3      # Probability of update at each check
    min_episodes_before_update = 50  # Minimum episodes before first update
    last_update_episode = start_episode  # Track when the last update happened
    
    # Initialize scaler for mixed precision training
    scaler = GradScaler()

    print(f"Continuing training from episode {start_episode}")
    for episode in range(start_episode, start_episode + episodes):
        obs, legal_actions = env.reset()
        done = False
        total_reward = 0
        
        # Track whose turn it is (0 = main agent, 1 = opponent)
        current_player = 0 if env.env.game.get_player_id() == 0 else 1
        
        while not done:
            if current_player == 0:  # Main agent's turn
                # Choose action using epsilon-greedy
                action = main_agent.act(obs, legal_actions, epsilon)
                
                # Take action in environment
                next_obs, reward, done = env.step(action)
                total_reward += reward
                
                # Store transition in replay buffer
                next_state = next_obs[0] if next_obs[0] is not None else np.zeros_like(obs)
                replay_buffer.push(obs, action, reward, next_state, done)
                
                # Update observation
                obs = next_obs[0] if next_obs[0] is not None else None
                legal_actions = next_obs[1] if next_obs[1] is not None else []
                
                if obs is None:
                    break
                
                # Update current player
                current_player = 1 if env.env.game.get_player_id() == 1 else 0
                
            else:  # Opponent's turn
                # Choose action using opponent policy
                action = opponent_agent.act(obs, legal_actions, opponent_epsilon)
                
                # Take action in environment
                next_obs, reward, done = env.step(action)
                
                # Update observation
                obs = next_obs[0] if next_obs[0] is not None else None
                legal_actions = next_obs[1] if next_obs[1] is not None else []
                
                if obs is None:
                    break
                
                # Update current player
                current_player = 0 if env.env.game.get_player_id() == 0 else 1
            
            # Train the main agent
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors without pin_memory
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)
                
                # Use mixed precision training
                with autocast():
                    q_values = main_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = main_model(next_states).argmax(1)
                        next_q_values = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        target_q_values = rewards + gamma * next_q_values * (1 - dones)
                    
                    loss = nn.MSELoss()(q_values, target_q_values)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        
        # Update target network periodically
        if episode % 50 == 0:
            target_model.load_state_dict(main_model.state_dict())
            if episode % 500 == 0:
                print(f"Episode {episode}, Buffer Size: {len(replay_buffer)}")
            if len(replay_buffer) >= batch_size:
                print(f"Episode {episode}, Loss: {loss.item()}")
                print(f"Average Q-value: {q_values.mean().item():.4f}")
                print(f"Epsilon: {epsilon:.4f}")
        
        # Log episode results
        if episode % 500 == 0:
            log_episode(episode, total_reward)
        
        # Randomized opponent model updates
        episodes_since_update = episode - last_update_episode
        if episode > min_episodes_before_update and episodes_since_update >= base_update_freq:
            # Check with probability update_prob
            if np.random.random() < update_prob:
                print(f"Randomly updating opponent model at episode {episode}")
                opponent_model.load_state_dict(main_model.state_dict())
                last_update_episode = episode
                
                # Slightly increase update probability over time to ensure updates happen
                update_prob = min(0.5, update_prob + 0.01)
            else:
                # If we've gone too long without an update (3x base frequency), force an update
                if episodes_since_update >= 3 * base_update_freq:
                    print(f"Forcing opponent model update at episode {episode} after {episodes_since_update} episodes")
                    opponent_model.load_state_dict(main_model.state_dict())
                    last_update_episode = episode
        
        # Save model periodically
        if episode % 2000 == 0:
            torch.save(main_model.state_dict(), f"model_episode_{episode}.pth")
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

if __name__ == "__main__":
    try:
        train_with_self_play(
            pretrained_model_path="model_episode_108000.pth",
            start_episode=108000,
            epsilon_start=0.2  # Lower epsilon since we're continuing training
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc() 