# test_model.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import PokerEnv
from model import init_model
import rlcard
import os
import glob

# Action mapping for No-Limit Hold'em (based on your legal actions)
ACTION_MAP = {
    0: "fold",
    1: "call",
    2: "raise (small)",  # Not seen in your output, but included for completeness
    3: "check",
    4: "raise (large)"   # Adjust based on your env's action space
}

# Function to visualize Q-values
def plot_q_values(q_values, legal_actions, action_chosen, step):
    #plt.clf()  # Clear previous plot
    actions = range(len(q_values))  # All possible actions (0 to num_actions-1)
    q_values_np = q_values.cpu().numpy()
    
    # Plot all Q-values, highlight legal actions and chosen action
    '''
    plt.bar(actions, q_values_np, color='lightgray', label='Q-values')
    plt.bar(legal_actions, q_values_np[legal_actions], color='blue', label='Legal Actions')
    plt.bar(action_chosen, q_values_np[action_chosen], color='red', label='Chosen Action')
    
    plt.xlabel('Actions')
    plt.ylabel('Q-Value')
    plt.title(f'Q-Values at Step {step}')
    plt.legend()
    plt.pause(1)  # Pause to display the plot briefly
    '''

def load_model_with_compatibility(model_path, input_size, output_size, device):
    """Load a model with compatibility handling for architecture changes"""
    print(f"Loading model: {model_path}")
    model = init_model(input_size, output_size, device)
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if it's a direct state dict or wrapped in a dictionary
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            print("Found model_state_dict in loaded file")
            state_dict = state_dict['model_state_dict']
        
        # Filter out unexpected keys
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # Update the model dict with filtered state dict
        model_dict.update(filtered_state_dict)
        
        # Load the filtered state dict
        model.load_state_dict(model_dict, strict=False)
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    return model

def test_model(model_path="model_episode_90000.pth", episodes=5):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment and model
    env = PokerEnv()
    model = load_model_with_compatibility(model_path, input_size=54, output_size=env.num_actions, device=device)
    model.eval()  # Set to evaluation mode
    model.to(device)

    # For rendering game state
    rlcard_env = env.env  # Access the underlying RLCard environment

    for episode in range(episodes):
        obs, legal_actions = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\n=== Episode {episode} ===")
        print(f"Initial state: Player {rlcard_env.game.get_player_id()}")
        
        while not done:
            # Convert observation to tensor - add batch dimension
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get Q-values from model
            with torch.no_grad():
                q_values = model(obs_tensor).squeeze(0)  # Remove batch dimension after forward pass
            
            # Filter Q-values for legal actions and choose the best
            legal_q_values = q_values[legal_actions]
            action = legal_actions[torch.argmax(legal_q_values).item()]
            
            # Display game state directly from rlcard_env.game
            game = rlcard_env.game
            player_id = game.get_player_id()
            state = game.get_state(player_id)  # Get the state for the current player
            
            # Convert cards to string representation
            hand = [str(card) for card in game.players[player_id].hand]
            public_cards = [str(card) for card in game.public_cards]
            
            # Get chip information
            chips_in_pot = [player.in_chips for player in game.players]  # Chips contributed this round
            total_pot = game.dealer.pot  # Total pot size
            
            print(f"\nStep {step}:")
            #print(f"Observation: {obs}")
            print(f"Legal Actions: {legal_actions}")
            print(f"Community Cards: {public_cards}")
            print(f"Player {player_id} Hand: {hand}")
            print(f"Chips in Pot (per player): {chips_in_pot}")
            print(f"Total Pot: {total_pot}")
            print(f"Q-Values: {q_values}")
            print(f"Chosen Action: {action} ({ACTION_MAP.get(action, 'unknown')})")

            # Visualize Q-values
            plot_q_values(q_values, legal_actions, action, step)
            
            # Take action in environment
            next_obs, reward, done = env.step(action)
            total_reward += reward
            
            # Update observation
            obs = next_obs[0] if next_obs[0] is not None else None
            legal_actions = next_obs[1] if next_obs[1] is not None else []
            
            if obs is None:
                print("Game ended (obs is None)")
                break
            
            step += 1
        
        # Episode summary
        print(f"Episode {episode} finished. Total Reward: {total_reward}")
    
    #plt.close()  # Close the plot window after testing

if __name__ == "__main__":
    # Import os module if not already imported
    import os
    import glob
    
    # Specify the exact directory path
    model_dir = r"C:\Users\ismas\Poker Machine learning\Poker-Machine-Learning"
    
    try:
        print(f"Checking directory: {model_dir}")
        
        # Check if directory exists
        if not os.path.exists(model_dir):
            print(f"ERROR: Directory does not exist: {model_dir}")
            # Try to list parent directory contents to debug
            parent_dir = os.path.dirname(model_dir)
            if os.path.exists(parent_dir):
                print(f"Parent directory exists: {parent_dir}")
                print(f"Contents of parent directory: {os.listdir(parent_dir)}")
            
            # Fall back to default
            print("Using default model path")
            test_model(model_path="model_episode_110000.pth", episodes=20)
        else:
            # List all files in the directory
            all_files = os.listdir(model_dir)
            print(f"All files in directory ({len(all_files)} files):")
            for file in all_files:
                print(f"  - {file}")
            
            # Filter for model files
            model_files = [f for f in all_files if f.startswith("model_episode_") and f.endswith(".pth")]
            
            if not model_files:
                print(f"No model files found in {model_dir}")
                # Fall back to default
                print("Using default model path")
                test_model(model_path="model_episode_110000.pth", episodes=20)
            else:
                # Sort by episode number to find the latest
                model_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
                print(f"Found {len(model_files)} model files:")
                for i, model in enumerate(model_files):
                    print(f"  {i+1}. {model} (Episode {model.split('_')[2].split('.')[0]})")
                
                latest_model = model_files[0]
                model_path = os.path.join(model_dir, latest_model)
                print(f"Using latest model: {model_path} (Episode {latest_model.split('_')[2].split('.')[0]})")
                test_model(model_path=model_path, episodes=20)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()