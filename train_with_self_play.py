import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from environment import PokerEnv
from model import init_model, load_model_with_compatibility
from replay_buffer import ReplayBuffer
from utils import log_episode
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import random
from collections import deque

def train_with_self_play(episodes=222001, batch_size=2048, gamma=0.97, epsilon_start=0.3, 
                         epsilon_end=0.1, epsilon_decay=0.9995, start_episode=0,
                         pretrained_model_path=None, model_save_dir="Poker-Machine-Learning\\Poker-Machine-Learning"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        cudnn.benchmark = True
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    else:
        print("Training on CPU")
    
    env = PokerEnv()
    
    # Initialize a new model if no pretrained model is provided
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        main_model = load_model_with_compatibility(pretrained_model_path, 
                                                  input_size=54, 
                                                  output_size=env.num_actions, 
                                                  device=device)
    else:
        print("No pretrained model found. Initializing a new model.")
        main_model = init_model(input_size=54, output_size=env.num_actions, device=device)
        start_episode = 0  # Reset start_episode to 0 when starting fresh
    
    target_model = init_model(input_size=54, output_size=env.num_actions, device=device)
    target_model.load_state_dict(main_model.state_dict())
    opponent_model = init_model(input_size=54, output_size=env.num_actions, device=device)
    opponent_model.load_state_dict(main_model.state_dict())
    
    # Historical self-play setup
    snapshot_dir = "Poker-Machine-Learning"
    os.makedirs(snapshot_dir, exist_ok=True)  # Create directory if it doesn't exist
    snapshot_interval = 1000  # Save every 1000 episodes
    max_snapshots = 10  # Keep up to 10 past models
    snapshot_pool = deque(maxlen=max_snapshots)  # FIFO queue for snapshots
    
    # Initialize pool with current model if starting fresh
    initial_snapshot = os.path.join(snapshot_dir, f"model_episode_{start_episode}.pth")
    torch.save(main_model.state_dict(), initial_snapshot)
    snapshot_pool.append(initial_snapshot)
    
    # Initialize optimizer, loss function, and replay buffer
    optimizer = optim.Adam(main_model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Initialize training variables
    epsilon = epsilon_start
    target_update_frequency = 1000
    
    # Simplify GradScaler initialization to avoid version issues
    scaler = GradScaler()
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Training loop
    total_rewards = []
    for episode in range(start_episode + 1, episodes + 1):
        # Occasionally use a historical model as the opponent
        if episode % 50 == 0 and len(snapshot_pool) > 1:
            # 50% chance to use a historical model
            if random.random() < 0.5:
                snapshot_path = random.choice(list(snapshot_pool)[:-1])  # Don't pick the most recent
                opponent_model.load_state_dict(torch.load(snapshot_path))
                print(f"Using historical model from {snapshot_path} as opponent")
            else:
                # Use current model as opponent
                opponent_model.load_state_dict(main_model.state_dict())
        else:
            # Use current model as opponent
            opponent_model.load_state_dict(main_model.state_dict())
        
        # Reset environment
        state, legal_actions = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        
        # Play one episode
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(legal_actions)
            else:
                with torch.no_grad():
                    main_model.eval()  # Set model to evaluation mode
                    q_values = main_model(state)
                    main_model.train()  # Set model back to training mode
                    # Filter q-values for legal actions only
                    legal_q_values = {a: q_values[0][a].item() for a in legal_actions}
                    action = max(legal_q_values, key=legal_q_values.get)
            
            # Take action in environment
            next_state_info, reward, done = env.step(action)
            
            # Process next state
            if done:
                # For terminal states, store a zero array instead of None
                next_state_array = np.zeros_like(state.cpu().numpy()[0])
                next_state = None  # Explicitly set to None for clarity
                next_legal_actions = []
            else:
                next_state, next_legal_actions = next_state_info
                next_state_array = next_state
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            
            # Store transition in replay buffer
            replay_buffer.push(state.cpu().numpy()[0], action, reward, 
                              next_state_array, done)
            
            # Move to next state
            if not done:  # Only update state if not done
                state = next_state
                legal_actions = next_legal_actions
            episode_reward += reward
            
            # Train the network if enough samples
            if len(replay_buffer) > batch_size:
                # Sample from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # Compute next Q values using target network (outside of autocast)
                with torch.no_grad():
                    target_model.eval()
                    next_q_values = torch.zeros(batch_size, device=device)
                    if (~dones.bool()).any():
                        next_state_values = target_model(next_states).max(1)[0]
                        next_q_values[~dones.bool()] = next_state_values[~dones.bool()]
                    target_model.train()
                
                # Compute target Q values (outside of autocast)
                target_q_values = rewards + gamma * next_q_values
                
                # Compute current Q values and loss (with autocast)
                try:
                    autocast_context = torch.amp.autocast('cuda')
                except TypeError:
                    autocast_context = autocast()
                
                with autocast_context:
                    current_q_values = main_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    loss = criterion(current_q_values, target_q_values)
                
                # Optimize the model
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Update target network
        if episode % target_update_frequency == 0:
            target_model.load_state_dict(main_model.state_dict())
        
        # Save model snapshot for self-play
        if episode % snapshot_interval == 0:
            snapshot_path = os.path.join(snapshot_dir, f"model_episode_{episode}.pth")
            torch.save(main_model.state_dict(), snapshot_path)
            snapshot_pool.append(snapshot_path)
            
            # Also save as latest model
            torch.save({
                'episode': episode,
                'model_state_dict': main_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }, os.path.join(model_save_dir, "latest_model.pth"))
        
        # Log progress
        total_rewards.append(episode_reward)
        if episode % 1000 == 0:
            avg_reward = sum(total_rewards[-100:]) / min(100, len(total_rewards))
            log_episode(episode, avg_reward)
            print(f"Epsilon: {epsilon:.4f}")
    
    # Save final model
    torch.save({
        'episode': episodes,
        'model_state_dict': main_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
    }, os.path.join(model_save_dir, "final_model.pth"))
    
    return main_model

if __name__ == "__main__":
    try:
        # Look for existing model in the correct directory
        model_dir = "Poker-Machine-Learning"
        pretrained_path = os.path.join(model_dir, "latest_model.pth")
        
        if os.path.exists(pretrained_path):
            print(f"Found existing model at {pretrained_path}")
            # Load the existing model and continue training
            train_with_self_play(
                pretrained_model_path=pretrained_path,
                start_episode=0,  # This will be updated from the model file
                epsilon_start=0.3,
                model_save_dir=model_dir
            )
        else:
            print(f"No existing model found at {pretrained_path}")
            # Start training from scratch
            train_with_self_play(
                pretrained_model_path=None,
                start_episode=0,
                epsilon_start=0.3,
                model_save_dir=model_dir
            )
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()