#train_with_self_play.py
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
import gc  # Add garbage collection
import time
import signal
from datetime import datetime

# Force CUDA to use the NVIDIA GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add a watchdog timer to detect and handle stalls
class TrainingWatchdog:
    def __init__(self, timeout=1800):  # 30 minutes timeout
        self.timeout = timeout
        self.last_episode_time = time.time()
        
    def update(self):
        self.last_episode_time = time.time()
        
    def check(self):
        elapsed = time.time() - self.last_episode_time
        return elapsed < self.timeout
        
    def time_since_update(self):
        return time.time() - self.last_episode_time

# Move the cpu_optimizer_state function to module level so it's accessible for emergency checkpoints
def cpu_optimizer_state(state_dict):
    cpu_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.cpu()
        elif isinstance(v, dict):
            cpu_state[k] = cpu_optimizer_state(v)  # Recursive call for nested dicts
        else:
            cpu_state[k] = v
    return cpu_state

def train_with_self_play(episodes=2220001, batch_size=8192, gamma=0.97, epsilon_start=0.3, 
                         epsilon_end=0.1, epsilon_decay=0.9997805, start_episode=0,
                         pretrained_model_path=None, model_save_dir="Poker-Machine-Learning\\Poker-Machine-Learning",
                         resume_from_checkpoint=False):
    # Setup watchdog timer to detect stalls
    watchdog = TrainingWatchdog(timeout=1800)  # 30 minutes timeout
    
    # Create a checkpoint directory
    checkpoint_dir = os.path.join(model_save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device with stronger memory management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        cudnn.benchmark = True
        torch.cuda.empty_cache()  # Clear cache before starting
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        torch.cuda.set_device(0)  # Explicitly set to first GPU
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    else:
        print("Training on CPU")
    
    env = PokerEnv()
    
    # Get actual state dimension by checking the environment
    state_sample, _ = env.reset()
    input_size = len(state_sample)  # Dynamically determine the input size
    print(f"Detected state dimension: {input_size}")
    
    # Initialize a new model if no pretrained model is provided
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        main_model = load_model_with_compatibility(pretrained_model_path, 
                                                  input_size=input_size,  # Use actual input size 
                                                  output_size=env.num_actions, 
                                                  device=device)
    else:
        print("No pretrained model found. Initializing a new model.")
        main_model = init_model(input_size=input_size, output_size=env.num_actions, device=device)
        start_episode = 0  # Reset start_episode to 0 when starting fresh
    
    target_model = init_model(input_size=input_size, output_size=env.num_actions, device=device)
    target_model.load_state_dict(main_model.state_dict())
    opponent_model = init_model(input_size=input_size, output_size=env.num_actions, device=device)
    opponent_model.load_state_dict(main_model.state_dict())
    
    # Historical self-play setup, will find all existing models in folder
    snapshot_dir = model_save_dir
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_interval = 1000
    max_snapshots = 10  # This will limit newly created models
    
    # Find all existing model snapshots in the directory
    snapshot_pool = []
    existing_snapshots = [f for f in os.listdir(snapshot_dir) if f.startswith('model_episode_') and f.endswith('.pth')]
    
    if existing_snapshots:
        # Sort snapshots by episode number
        existing_snapshots.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        for snapshot_file in existing_snapshots:
            snapshot_path = os.path.join(snapshot_dir, snapshot_file)
            if os.path.exists(snapshot_path):
                snapshot_pool.append(snapshot_path)
                print(f"Added existing snapshot: {snapshot_path}")
        
        print(f"Found {len(snapshot_pool)} existing model snapshots for opponent selection")
    else:
        # If no snapshots exist, create the initial one
        initial_snapshot = os.path.join(snapshot_dir, f"model_episode_{start_episode}.pth")
        torch.save(main_model.state_dict(), initial_snapshot)
        snapshot_pool.append(initial_snapshot)
        print(f"Created initial snapshot: {initial_snapshot}")
    
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
    reward_window_size = 500  # How many episodes to consider for trend
    reward_stagnation_threshold = 0.005  # Threshold for considering rewards stagnated
    epsilon_boost = 0.05  # How much to boost epsilon when stagnation is detected
    boost_cooldown = 1000  # Minimum episodes between epsilon boosts
    last_boost_episode = 0
    reward_history = deque(maxlen=reward_window_size*2)  # Store more history for better trend analysis
    
    try:
        # Add checkpoint recovery
        if resume_from_checkpoint:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                print(f"Resuming from checkpoint: {checkpoint_path}")
                
                checkpoint = torch.load(checkpoint_path)
                start_episode = checkpoint['episode']
                epsilon = checkpoint['epsilon']
                
                main_model = load_model_with_compatibility(checkpoint_path, 
                                                        input_size=input_size, 
                                                        output_size=env.num_actions, 
                                                        device=device)
                
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Rebuild replay buffer (this will be empty, but structure will be correct)
                replay_buffer = ReplayBuffer(capacity=100000)
                
                # If you stored the rewards history
                if 'total_rewards' in checkpoint:
                    total_rewards = checkpoint['total_rewards']
                else:
                    total_rewards = []
                
                print(f"Resumed training from episode {start_episode}")
            else:
                print("No checkpoints found. Starting from scratch.")
                # ... normal initialization code ...
        
        # Training loop (modified)
        for episode in range(start_episode + 1, episodes + 1):
            watchdog.update()  # Reset watchdog timer at the start of each episode
            
            # Occasionally use a historical model as the opponent
            if episode % 100 == 0 and len(snapshot_pool) > 1: 
                # 70% chance to use a historical model
                if random.random() < 0.7:
                    # Select a model with probability weighted toward older models for diversity
                    weights = [1.0/(i+1) for i in range(len(snapshot_pool))]
                    total_weight = sum(weights)
                    weights = [w/total_weight for w in weights]
                    
                    # Don't exclude the most recent model - it's also useful for training
                    snapshot_path = random.choices(snapshot_pool, weights=weights, k=1)[0]
                    
                    try:
                        opponent_model.load_state_dict(torch.load(snapshot_path, map_location=device))
                        print(f"Using historical model from {snapshot_path} as opponent")
                    except Exception as e:
                        print(f"Error loading model {snapshot_path}: {e}. Using current model instead.")
                        opponent_model.load_state_dict(main_model.state_dict())
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
                
                # Converts to tensors for more efficient memory usage
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # Process in smaller chunks if needed for very large batches
                chunk_size = 4096  # Process 4K samples at a time
                loss_sum = 0
                td_errors = []  # To store TD errors for logging

                for i in range(0, batch_size, chunk_size):
                    end = min(i + chunk_size, batch_size)
                    chunk_states = states[i:end]
                    chunk_actions = actions[i:end]
                    chunk_rewards = rewards[i:end]
                    chunk_next_states = next_states[i:end]
                    chunk_dones = dones[i:end]
                    
                    with torch.no_grad():
                        # Double DQN: use main network for action selection, target network for value estimation
                        next_actions = main_model(chunk_next_states).argmax(1)
                        next_q_values = target_model(chunk_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        # Apply non-terminal mask directly in the target calculation
                        target_q_values = chunk_rewards + gamma * next_q_values * (~chunk_dones.bool())
                    
                    try:
                        autocast_context = torch.amp.autocast('cuda')
                    except TypeError:
                        autocast_context = autocast()
                    
                    with autocast_context:
                        current_q_values = main_model(chunk_states).gather(1, chunk_actions.unsqueeze(1)).squeeze(1)
                        loss = criterion(current_q_values, target_q_values)
                        loss_sum += loss.item()
                        
                        # Compute TD Error for this chunk
                        td_error = torch.abs(target_q_values - current_q_values).mean().item()
                        td_errors.append(td_error)
                    
                    # Optimize the model
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                # Average TD Error across all chunks
                avg_td_error = sum(td_errors) / len(td_errors)
                
                # Clean up to prevent memory fragmentation
                del states, actions, rewards, next_states, dones
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Update epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Update target network
            if episode % target_update_frequency == 0:
                target_model.load_state_dict(main_model.state_dict())
            
            # Save model snapshot for self-play
            if episode % snapshot_interval == 0:
                # Delete previous model objects before creating new ones
                if 'snapshot_model' in locals():
                    del snapshot_model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # Create a new model instance for the snapshot
                snapshot_model = init_model(input_size=input_size, output_size=env.num_actions, device=torch.device('cpu'))
                snapshot_model.load_state_dict({k: v.cpu() for k, v in main_model.state_dict().items()})
                
                snapshot_path = os.path.join(snapshot_dir, f"model_episode_{episode}.pth")
                torch.save(snapshot_model.state_dict(), snapshot_path)
                snapshot_pool.append(snapshot_path)
                
                # Print summary of available snapshots
                print(f"Added new snapshot: {snapshot_path}")
                print(f"Total snapshots available: {len(snapshot_pool)}")
                
                # Limit the total number of snapshots used for memory reasons
                if len(snapshot_pool) > 50:  # Allow up to 50 for more diversity
                    # Only remove snapshots that follow the pattern model_episode_X.pth where X is a multiple of 1000
                    # This preserves milestone snapshots
                    removable_snapshots = [s for s in snapshot_pool if int(s.split('_')[2].split('.')[0]) % 5000 != 0]
                    if len(removable_snapshots) > 40:  # If we have too many non-milestone snapshots
                        # Remove some of the middle snapshots to maintain diversity
                        removable_snapshots.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                        to_remove = removable_snapshots[5:-5]  # Keep oldest 5 and newest 5
                        
                        for i, old_path in enumerate(to_remove):
                            if i % 2 == 0:  # Remove every other one to maintain spacing
                                if old_path in snapshot_pool:
                                    snapshot_pool.remove(old_path)
                                    print(f"Removed snapshot from pool: {old_path}")
                
                # Modified version that checks if value is a tensor first
                # Also save as latest model
                torch.save({
                    'episode': episode,
                    'model_state_dict': snapshot_model.state_dict(),  # Use CPU model
                    'optimizer_state_dict': cpu_optimizer_state(optimizer.state_dict()),
                    'epsilon': epsilon,
                }, os.path.join(model_save_dir, "latest_model.pth"))
                
                # Clean up
                del snapshot_model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"GPU Memory after snapshot: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
            # Log progress
            total_rewards.append(episode_reward)
            reward_history.append(episode_reward)
            if episode % 100 == 0:
                avg_reward = sum(total_rewards[-100:]) / min(100, len(total_rewards))
                # Modified to log TD error separately instead of as a parameter
                log_episode(episode, avg_reward)
                if 'avg_td_error' in locals():
                    print(f"Epsilon: {epsilon:.4f}, Avg TD Error: {avg_td_error:.4f}")
                else:
                    print(f"Epsilon: {epsilon:.4f}")
                if episode % 500 == 0:
                    print(f"Snapshots available: {len(snapshot_pool)}")
            
            # Save more frequent checkpoints
            if episode % 200 == 0:  # More frequent checkpoints
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
                torch.save({
                    'episode': episode,
                    'model_state_dict': main_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epsilon': epsilon,
                    'total_rewards': total_rewards[-1000:] if len(total_rewards) > 0 else [],
                }, checkpoint_path)
                
                # Keep only the 5 most recent checkpoints
                checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')], 
                                       key=lambda x: int(x.split('_')[1].split('.')[0]))
                if len(checkpoint_files) > 5:
                    for old_file in checkpoint_files[:-5]:
                        os.remove(os.path.join(checkpoint_dir, old_file))
            
            # Add watchdog checking
            if episode % 10 == 0:
                if not watchdog.check():
                    print(f"WARNING: Training seems stalled ({watchdog.time_since_update():.1f} seconds since last update)")
                    # Force save and raise exception to trigger auto-recovery
                    emergency_path = os.path.join(checkpoint_dir, f"emergency_{episode}.pth")
                    torch.save({
                        'episode': episode,
                        'model_state_dict': main_model.state_dict(),
                        'optimizer_state_dict': cpu_optimizer_state(optimizer.state_dict()),
                        'epsilon': epsilon,
                        'total_rewards': total_rewards[-1000:] if len(total_rewards) > 0 else [],
                    }, emergency_path)
                    raise TimeoutError("Training watchdog detected stall")
                    
            # Check for reward stagnation or decline every reward_window_size episodes
            if episode % reward_window_size == 0 and episode > reward_window_size*2 and episode - last_boost_episode > boost_cooldown:
                # Get recent and previous window averages
                recent_avg = sum(list(reward_history)[-reward_window_size:]) / reward_window_size
                previous_avg = sum(list(reward_history)[-2*reward_window_size:-reward_window_size]) / reward_window_size
                
                # Check if rewards are stagnating or declining
                relative_change = (recent_avg - previous_avg) / (abs(previous_avg) + 1e-8)  # Avoid division by zero
                
                if relative_change < reward_stagnation_threshold:
                    # Boost epsilon to encourage more exploration
                    old_epsilon = epsilon
                    epsilon = min(epsilon_start, epsilon + epsilon_boost)  # Increase but don't exceed starting epsilon
                    last_boost_episode = episode
                    print(f"Episode {episode}: Reward stagnation detected. Boosting epsilon from {old_epsilon:.4f} to {epsilon:.4f}")
            
    except Exception as e:
        print(f"Exception during training: {e}")
        # Save emergency checkpoint
        emergency_path = os.path.join(checkpoint_dir, f"emergency_{start_episode + 1}.pth")
        try:
            torch.save({
                'episode': episode if 'episode' in locals() else start_episode,
                'model_state_dict': main_model.state_dict() if 'main_model' in locals() else None,
                'optimizer_state_dict': cpu_optimizer_state(optimizer.state_dict()) if 'optimizer' in locals() else None,
                'epsilon': epsilon if 'epsilon' in locals() else epsilon_start,
                'total_rewards': total_rewards[-1000:] if 'total_rewards' in locals() and len(total_rewards) > 0 else [],
            }, emergency_path)
            print(f"Emergency checkpoint saved to {emergency_path}")
        except Exception as save_error:
            print(f"Failed to save emergency checkpoint: {save_error}")
        
        raise e
    
    return main_model

if __name__ == "__main__":
    # Set up auto-recovery flag
    auto_recovery = False
    recovery_attempts = 0
    max_recovery_attempts = 3
    
    while recovery_attempts <= max_recovery_attempts:
        try:
            # Look for existing model or checkpoint
            model_dir = "Poker-Machine-Learning"
            checkpoint_dir = os.path.join(model_dir, "checkpoints")
            pretrained_path = os.path.join(model_dir, "latest_model.pth")
            
            # Check for checkpoints first if in recovery mode
            if auto_recovery and os.path.exists(checkpoint_dir):
                print(f"Attempting auto-recovery (attempt {recovery_attempts}/{max_recovery_attempts})")
                train_with_self_play(
                    pretrained_model_path=None,  # We'll load from checkpoint instead
                    start_episode=0,  # Will be updated from checkpoint
                    epsilon_start=0.3,
                    model_save_dir=model_dir,
                    batch_size=8192,  # Further reduced for stability
                    resume_from_checkpoint=True  # Enable checkpoint loading
                )
            elif os.path.exists(pretrained_path):
                print(f"Found existing model at {pretrained_path}")
                train_with_self_play(
                    pretrained_model_path=pretrained_path,
                    start_episode=0,
                    epsilon_start=0.3,
                    model_save_dir=model_dir,
                    batch_size=8192  # Reduced further from 16384
                )
            else:
                print(f"No existing model found at {pretrained_path}")
                train_with_self_play(
                    pretrained_model_path=None,
                    start_episode=0,
                    epsilon_start=0.3,
                    model_save_dir=model_dir,
                    batch_size=8192
                )
            
            # If we get here, training completed successfully
            break
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Set up for auto-recovery on next attempt
            auto_recovery = True
            recovery_attempts += 1
            
            if recovery_attempts <= max_recovery_attempts:
                print(f"Will attempt auto-recovery in 30 seconds...")
                time.sleep(30)  # Wait before retrying
            else:
                print("Maximum recovery attempts reached. Giving up.")