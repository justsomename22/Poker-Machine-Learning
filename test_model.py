# test_model.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import PokerEnv
from model import init_model
import rlcard
import os
import glob
import datetime
import random

# Create a logging function that writes to both console and file
class Logger:
    def __init__(self, log_file_path):
        # Add UTF-8 encoding to handle emoji characters
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        print(f"Logging output to: {log_file_path}")
    
    def log(self, message):
        print(message)
        self.log_file.write(f"{message}\n")
        self.log_file.flush()  # Ensure log is written immediately
    
    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# Create a log file with timestamp
def create_log_file():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Use a fixed filename "poker_debug.log" instead of a timestamped one
    return os.path.join(log_dir, "poker_debug.log")

# Action mapping for No-Limit Hold'em (based on your legal actions)
ACTION_MAP = {
    0: "Fold",
    1: "Call",
    2: "Raise 1/3 pot",
    3: "Raise 1/2 pot",
    4: "Raise full pot",
    5: "Check",
    6: "All-in"
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
    logger.log(f"Loading model: {model_path}")
    model = init_model(input_size, output_size, device)
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if it's a direct state dict or wrapped in a dictionary
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            logger.log("Found model_state_dict in loaded file")
            state_dict = state_dict['model_state_dict']
        
        # Filter out unexpected keys
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # Update the model dict with filtered state dict
        model_dict.update(filtered_state_dict)
        
        # Load the filtered state dict
        model.load_state_dict(model_dict, strict=False)
        logger.log(f"Successfully loaded model: {model_path}")
    except Exception as e:
        logger.log(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    return model

# Function to explain reward calculations based on action and game state
def explain_reward(env, action, reward, state, equity, pot_odds):
    """Provide an explanation for why a particular reward was given."""
    explanation = f"Reward: {reward:.4f} | "
    
    # Get current game state information
    game = env.env.game
    pot_size = game.dealer.pot
    player_id = env.player_id
    street = env.street
    
    # Determine which street we're on
    street_names = ["Preflop", "Flop", "Turn", "River"]
    current_street = street_names[min(street, 3)]
    
    if reward > 0:
        explanation += f"Positive reward on {current_street}. "
        
        # Check if this was a terminal reward (hand ended)
        if game.is_over():
            explanation += "Hand complete. "
            # Check if opponent folded
            opponent_id = 1 - player_id
            if game.players[opponent_id].status == 'folded':
                explanation += f"Opponent folded. Your equity was {equity:.2f}. "
                if equity < 0.4:
                    explanation += "Good bluff! "
                else:
                    explanation += "Value bet paid off. "
            else:
                explanation += f"Won at showdown with equity {equity:.2f}. "
        else:
            # Intermediate reward
            if action == 5:  # Check
                explanation += "Reward for checking with medium strength hand. "
            elif action in [2, 3, 4]:  # Raises
                explanation += f"Reward for raising with equity {equity:.2f} vs pot odds {pot_odds:.2f}. "
            elif action == 1:  # Call
                explanation += f"Reward for calling with sufficient equity ({equity:.2f}) vs pot odds ({pot_odds:.2f}). "
    
    elif reward < 0:
        explanation += f"Negative reward on {current_street}. "
        
        if game.is_over():
            explanation += "Hand complete. "
            if action == 0:  # Fold
                explanation += f"Folded with equity {equity:.2f} vs pot odds {pot_odds:.2f}. "
                if equity > pot_odds + 0.1:
                    explanation += "Consider calling instead of folding here. "
            else:
                explanation += f"Lost with equity {equity:.2f}. "
                if equity < pot_odds:
                    explanation += "Calling with insufficient equity. "
        else:
            # Intermediate negative reward
            if action == 1 and equity < pot_odds - 0.05:
                explanation += "Penalty for calling with insufficient equity. "
            elif action in [2, 3, 4] and equity < 0.3:
                explanation += "Penalty for raising with weak hand. "
    
    else:  # reward == 0
        explanation += "Neutral action. "
    
    return explanation

# Track the true player hands across perspective changes
class HandTracker:
    def __init__(self):
        self.player0_hand = None
        self.player1_hand = None
        self.generated_cards = False
        self.reset()
    
    def reset(self):
        self.player0_hand = None
        self.player1_hand = None
        self.generated_cards = False
    
    def update_hand(self, player_id, hand):
        hand_str = [str(card) for card in hand]
        if player_id == 0:
            self.player0_hand = hand_str
        else:
            self.player1_hand = hand_str
        
        # After updating either hand, immediately check for duplicates
        self.ensure_unique_hands()
    
    def get_hand(self, player_id):
        # Ensure hands are unique before returning
        self.ensure_unique_hands()
        
        if player_id == 0:
            return self.player0_hand if self.player0_hand else ["Unknown", "Unknown"]
        else:
            return self.player1_hand if self.player1_hand else ["Unknown", "Unknown"]
    
    def ensure_unique_hands(self):
        """Make sure the hands are unique by generating cards if needed"""
        # Only proceed if we have both hands
        if not self.player0_hand or not self.player1_hand:
            return
        
        # Check if hands have any cards in common (impossible in real poker)
        has_common_cards = any(card in self.player1_hand for card in self.player0_hand)
        
        if has_common_cards or self.player0_hand == self.player1_hand:
            # Generate completely new hands for both players
            all_cards = self._generate_all_cards()
            
            # Start with the original player 0 hand
            player0_cards = self.player0_hand.copy() if self.player0_hand else []
            
            # For player 1, get completely different cards
            available_cards = [card for card in all_cards if card not in player0_cards]
            
            import random
            if len(available_cards) >= 2:
                self.player1_hand = random.sample(available_cards, 2)
                # Mark that we've generated cards
                self.generated_cards = True
    
    def _generate_all_cards(self):
        """Generate all possible cards in a deck"""
        all_cards = []
        suits = ['S', 'H', 'D', 'C']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        for rank in ranks:
            for suit in suits:
                all_cards.append(f"{rank}{suit}")
        
        return all_cards

# Add this function to get and display opponent's hole cards
def get_opponent_cards(env, hand_tracker, game):
    """Get the opponent's hole cards from the environment."""
    player_id = env.player_id
    opponent_id = 1 - player_id
    
    # Get the current player's hand first
    if hasattr(game, 'players') and len(game.players) > player_id:
        if game.players[player_id].hand:
            # Update the current player's hand
            hand_tracker.update_hand(player_id, game.players[player_id].hand)
    
    # Get the opponent's hand if available
    if hasattr(game, 'players') and len(game.players) > opponent_id:
        if game.players[opponent_id].hand:
            # Update the opponent's hand
            hand_tracker.update_hand(opponent_id, game.players[opponent_id].hand)
    
    # At this point, hand_tracker should have unique hands
    return hand_tracker.get_hand(opponent_id)

# Additional function to display human-readable suit symbols
def format_card(card_str):
    """Convert card strings to more readable format with suit symbols"""
    if len(card_str) != 2 or card_str in ["Unknown", "(Hidden)"]:
        return card_str
        
    rank, suit = card_str[0], card_str[1]
    
    # Use Unicode symbols for suits
    suit_symbols = {
        'S': '♠',
        'H': '♥',
        'D': '♦',
        'C': '♣'
    }
    
    # Format the card with colored suit if possible
    return f"{rank}{suit_symbols.get(suit, suit)}"

# Function to calculate actual bet amounts based on pot size and action type
def calculate_actual_bet_amount(action_id, official_bet_amount, total_pot, chips_in_pot, current_player_id, stacks):
    """Calculate the actual bet amount for display purposes based on action type and game state."""
    if action_id == 0:  # Fold
        return 0
    elif action_id == 1:  # Call
        return official_bet_amount  # Usually correct from environment
    elif action_id == 2:  # Raise 1/3 pot
        if official_bet_amount > 0:
            return official_bet_amount
        # Calculate 1/3 pot raise as fallback
        return max(1, int(total_pot / 3))
    elif action_id == 3:  # Raise 1/2 pot
        if official_bet_amount > 0:
            return official_bet_amount
        # Calculate 1/2 pot raise as fallback
        return max(1, int(total_pot / 2))
    elif action_id == 4:  # Raise full pot
        if official_bet_amount > 0:
            return official_bet_amount
        # Calculate full pot raise as fallback
        return max(1, total_pot)
    elif action_id == 5:  # Check
        return 0
    elif action_id == 6:  # All-in
        if official_bet_amount > 0:
            return official_bet_amount
        # Calculate all-in amount as fallback (player's remaining stack)
        return max(1, stacks[current_player_id])
    else:
        return official_bet_amount  # Default case

# Function to clarify bet amount descriptions with improved bet sizing information
def clarify_action_description(action_id, action_name, bet_amount, total_pot=None, chips_in_pot=None, current_player_id=None, stacks=None):
    """Provide a clearer description of actions with accurate bet sizing."""
    # Calculate actual bet amount if we have all the needed information
    if total_pot is not None and chips_in_pot is not None and current_player_id is not None and stacks is not None:
        actual_bet = calculate_actual_bet_amount(
            action_id, bet_amount, total_pot, chips_in_pot, current_player_id, stacks
        )
    else:
        actual_bet = bet_amount
        
    if action_id == 0:  # Fold
        return f"{action_name}"
    elif action_id == 1:  # Call
        if actual_bet == 0:
            return f"{action_name} (Check - no additional chips)"
        else:
            return f"{action_name} (Adding {actual_bet} chips to match bet)"
    elif action_id == 2:  # Raise 1/3 pot
        pot_info = f" from {total_pot} chip pot" if total_pot else ""
        return f"{action_name} (Betting {actual_bet} chips - 1/3 of pot{pot_info})"
    elif action_id == 3:  # Raise 1/2 pot
        pot_info = f" from {total_pot} chip pot" if total_pot else ""
        return f"{action_name} (Betting {actual_bet} chips - 1/2 of pot{pot_info})"
    elif action_id == 4:  # Raise full pot
        pot_info = f" from {total_pot} chip pot" if total_pot else ""
        return f"{action_name} (Betting {actual_bet} chips - full pot{pot_info})"
    elif action_id == 5:  # Check
        return f"{action_name} (No bet)"
    elif action_id == 6:  # All-in
        return f"{action_name} (Betting all {actual_bet} remaining chips)"
    else:
        return f"{action_name} (Amount: {actual_bet})"

# Class to implement AI opponent using the same model
class AIPlayer:
    def __init__(self, model, device, player_id):
        self.model = model
        self.device = device
        self.player_id = player_id
        self.name = f"AI Player {player_id}"
    
    def act(self, obs, legal_actions):
        """Choose an action given the current observation"""
        # Convert observation to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get Q-values from model
        with torch.no_grad():
            q_values = self.model(obs_tensor).squeeze(0)
        
        # Filter Q-values for legal actions and choose the best
        legal_q_values = {action: q_values[action].item() for action in legal_actions}
        action = max(legal_q_values, key=legal_q_values.get)
        
        return action, q_values

# Create a more aggressive AI player
class AggressiveAIPlayer(AIPlayer):
    def __init__(self, model, device, player_id, aggression_factor=0.2):
        super().__init__(model, device, player_id)
        self.aggression_factor = aggression_factor
        self.name = f"Aggressive AI {player_id}"
    
    def act(self, obs, legal_actions):
        """Choose an action with bias toward aggressive plays"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(obs_tensor).squeeze(0)
        
        # Add aggression bias to raise actions
        raise_actions = [2, 3, 4, 6]  # Raises and all-in
        for action in raise_actions:
            if action in legal_actions:
                q_values[action] += self.aggression_factor
        
        # Filter for legal actions and choose best
        legal_q_values = {action: q_values[action].item() for action in legal_actions}
        action = max(legal_q_values, key=legal_q_values.get)
        
        return action, q_values

# Create a deceptive AI player that occasionally makes unexpected moves
class DeceptiveAIPlayer(AIPlayer):
    def __init__(self, model, device, player_id, deception_rate=0.15):
        super().__init__(model, device, player_id)
        self.deception_rate = deception_rate
        self.name = f"Deceptive AI {player_id}"
    
    def act(self, obs, legal_actions):
        """Sometimes choose sub-optimal actions to be deceptive"""
        if random.random() < self.deception_rate and len(legal_actions) > 1:
            # Get model recommendation first
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.model(obs_tensor).squeeze(0)
            
            # Sort legal actions by Q-value (best to worst)
            legal_actions_sorted = sorted(
                legal_actions, 
                key=lambda a: q_values[a].item(), 
                reverse=True
            )
            
            # Choose second-best action instead of best
            action = legal_actions_sorted[1] if len(legal_actions_sorted) > 1 else legal_actions_sorted[0]
            return action, q_values
        else:
            # Use standard policy
            return super().act(obs, legal_actions)

# Update the test_model function for self-play
def test_model(model_path="model_episode_90000.pth", episodes=5, opponent_type="self"):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment
    env = PokerEnv()
    
    # Initialize hand tracker
    hand_tracker = HandTracker()
    
    # Get actual state dimension
    state_sample, _ = env.reset()
    input_size = len(state_sample)
    logger.log(f"Detected state dimension: {input_size}")
    
    # Load model
    model = load_model_with_compatibility(model_path, input_size=input_size, output_size=env.num_actions, device=device)
    model.eval()
    model.to(device)

    # Create AI players
    if opponent_type.lower() == "aggressive":
        logger.log(f"Creating aggressive opponent with higher tendency to raise")
        opponent = AggressiveAIPlayer(model, device, player_id=1, aggression_factor=0.2)
    elif opponent_type.lower() == "deceptive":
        logger.log(f"Creating deceptive opponent that sometimes makes unexpected moves")
        opponent = DeceptiveAIPlayer(model, device, player_id=1, deception_rate=0.15)
    else:  # "self" or default
        logger.log(f"Creating standard self-play opponent")
        opponent = AIPlayer(model, device, player_id=1)
    
    # Main player
    player = AIPlayer(model, device, player_id=0)
    
    # For rendering game state
    rlcard_env = env.env

    episode_results = []  # Track results for final summary
    
    for episode in range(episodes):
        obs, legal_actions = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Reset hand tracker for new episode
        hand_tracker.reset()
        
        logger.log(f"\n{'='*50}")
        logger.log(f"Episode {episode+1} of {episodes}".center(50))
        logger.log(f"{'='*50}")
        logger.log(f"Initial state: Player {rlcard_env.game.get_player_id()}")
        logger.log(f"Main player vs {opponent.name}")
        
        decisions = []  # Track decisions for episode summary
        
        while not done:
            # Get the current player's ID
            current_player_id = env.player_id
            
            # Determine which AI is playing
            current_ai = player if current_player_id == 0 else opponent
            
            # Get the action from the current AI
            action, q_values = current_ai.act(obs, legal_actions)
            
            # Display game state
            game = rlcard_env.game
            player_id = game.get_player_id()
            
            # Get cards
            hand = [str(card) for card in game.players[player_id].hand]
            opponent_hand = get_opponent_cards(env, hand_tracker, game)
            public_cards = [str(card) for card in game.public_cards]
            
            # Get chip information
            chips_in_pot = [player.in_chips for player in game.players]
            total_pot = game.dealer.pot
            stacks = env.player_stacks
            
            # Get street information
            street_names = ["Preflop", "Flop", "Turn", "River"]
            current_street = street_names[min(env.street, 3)]
            
            # Calculate equity and pot odds
            equity = env._calculate_equity(obs)
            pot_odds = env._calculate_pot_odds(obs)
            
            # Display detailed game state
            logger.log(f"\n{'#'*50}")
            logger.log(f"Step {step} - {current_street} - {current_ai.name}'s Turn".center(50))
            logger.log(f"{'#'*50}")
            
            # Format cards
            formatted_hand = [format_card(card) for card in hand]
            formatted_opponent_hand = [format_card(card) for card in opponent_hand]
            formatted_public_cards = [format_card(card) for card in public_cards]
            
            # Display hand information
            logger.log("\n📋 Game State:")
            logger.log(f"  🃏 Current Hand: {' '.join(formatted_hand)}")
            logger.log(f"  🎭 Opponent Hand: {' '.join(formatted_opponent_hand)}")
            if hand_tracker.generated_cards:
                logger.log(f"     (Note: Opponent hand was generated to avoid duplication)")
            logger.log(f"  🎲 Community Cards: {' '.join(formatted_public_cards) if formatted_public_cards else 'None'}")
            logger.log(f"  💰 Pot: {total_pot} | Current Player Chips in Pot: {chips_in_pot[player_id]} | Opponent Chips in Pot: {chips_in_pot[1-player_id]}")
            logger.log(f"  💹 Current Player Stack: {stacks[player_id]} | Opponent Stack: {stacks[1-player_id]}")
            logger.log(f"  📊 Current Player Equity: {equity:.2f} | Pot Odds: {pot_odds:.2f}")
            
            # Display legal actions with improved descriptions
            action_map = env._get_legal_actions()
            logger.log("\n🎮 Legal Actions:")
            for act in legal_actions:
                q_val = q_values[act].item()
                action_name = ACTION_MAP.get(act, f"Unknown({act})")
                bet_amount = action_map.get(act, 0)
                
                # Better description with accurate pot-based sizing
                clear_description = clarify_action_description(
                    act, action_name, bet_amount, total_pot, 
                    chips_in_pot, player_id, stacks
                )
                
                # Highlight chosen action
                is_best = (act == action)
                prefix = "→ " if is_best else "  "
                highlight = " (CHOSEN)" if is_best else ""
                
                logger.log(f"{prefix}{act}: {clear_description} - Q-value: {q_val:.4f}{highlight}")
            
            # Take action in environment
            next_obs, reward, done = env.step(action)
            total_reward += reward
            
            # Record this decision with additional information
            decisions.append({
                'player': current_ai.name,
                'street': current_street,
                'action': action,
                'action_name': ACTION_MAP.get(action, f"Unknown({action})"),
                'bet_amount': action_map.get(action, 0),
                'pot': total_pot,
                'chips_in_pot': chips_in_pot.copy() if chips_in_pot else None,
                'stacks': stacks.copy() if stacks else None,
                'player_id': player_id,
                'reward': reward
            })
            
            # Display chosen action with more emphasis
            action_name = ACTION_MAP.get(action, f"Unknown({action})")
            bet_amount = action_map.get(action, 0)
            clear_description = clarify_action_description(
                action, action_name, bet_amount, total_pot,
                chips_in_pot, player_id, stacks
            )
            logger.log(f"\n🎯 {current_ai.name} CHOSE: {action}: {clear_description}")
            logger.log(f"   This was the action with Q-value: {q_values[action].item():.4f}")
            
            # Provide reward explanation if main player
            if current_player_id == 0:
                reward_explanation = explain_reward(env, action, reward, obs, equity, pot_odds)
                logger.log(f"💡 {reward_explanation}")
            
            # Update for next step
            if not done:
                obs = next_obs[0]
                legal_actions = next_obs[1]
            else:
                # Show final game state
                final_state = "Won" if reward > 0 else "Lost" if reward < 0 else "Tied"
                current_player = "Main player" if current_player_id == 0 else opponent.name
                logger.log(f"\n🏁 Hand Complete: {current_player} {final_state} | Final Reward: {reward:.4f}")
                
                if hasattr(game, 'players'):
                    your_hand = [str(card) for card in game.players[0].hand] if game.players[0].hand else []
                    opp_hand = [str(card) for card in game.players[1].hand] if game.players[1].hand else []
                    logger.log(f"  Main Player's Final Hand: {' '.join([format_card(c) for c in your_hand])}")
                    logger.log(f"  {opponent.name}'s Final Hand: {' '.join([format_card(c) for c in opp_hand])}")
                    logger.log(f"  Community Cards: {' '.join([format_card(c) for c in [str(card) for card in game.public_cards]])}")
            
            step += 1
        
        # Episode summary
        episode_outcome = "WIN" if total_reward > 0 else "LOSS" if total_reward < 0 else "TIE"
        from_perspective = "from main player's perspective"
        episode_results.append(episode_outcome)
        
        logger.log(f"\n{'*'*50}")
        logger.log(f"Episode {episode+1} Summary".center(50))
        logger.log(f"{'*'*50}")
        logger.log(f"OUTCOME: {episode_outcome} {from_perspective}")
        logger.log(f"Total Reward: {total_reward:.4f}")
        logger.log(f"Total Steps: {step}")
        
        # Decision summary
        logger.log("\nDecision History:")
        for i, decision in enumerate(decisions):
            reward_str = f"Reward: {decision['reward']:.4f}" if decision['player'] == "AI Player 0" else ""
            if decision['reward'] > 0 and decision['player'] == "AI Player 0":
                reward_str += " ✓"
            elif decision['reward'] < 0 and decision['player'] == "AI Player 0":
                reward_str += " ✗"
            
            bet_amount = decision['bet_amount']
            # We don't have all the info for past decisions, but we have pot size
            action_desc = clarify_action_description(
                decision['action'], 
                decision['action_name'],
                bet_amount,
                decision['pot']
            )
            
            logger.log(f"  Step {i+1} ({decision['street']}): {decision['player']} - {action_desc} {reward_str}")
        
        logger.log("")
    
    # Final summary of all episodes
    logger.log(f"\n{'#'*50}")
    logger.log(f"OVERALL RESULTS".center(50))
    logger.log(f"{'#'*50}")
    logger.log(f"Episodes played: {episodes}")
    wins = episode_results.count("WIN")
    losses = episode_results.count("LOSS")
    ties = episode_results.count("TIE")
    logger.log(f"Main player wins: {wins} ({wins/episodes*100:.1f}%)")
    logger.log(f"Main player losses: {losses} ({losses/episodes*100:.1f}%)")
    logger.log(f"Ties: {ties} ({ties/episodes*100:.1f}%)")
    logger.log("")

if __name__ == "__main__":
    # Create log file
    log_file_path = create_log_file()
    logger = Logger(log_file_path)
    
    try:
        # Import os module if not already imported
        import os
        import glob
        
        # Specify the exact directory path
        model_dir = "Poker-Machine-Learning"
        
        logger.log(f"Checking directory: {model_dir}")
        
        # Check if directory exists
        if not os.path.exists(model_dir):
            logger.log(f"ERROR: Directory does not exist: {model_dir}")
            # Try fallback options:
            alternative_dirs = [
                ".",
                "./Poker-Machine-Learning",
                "../Poker-Machine-Learning"
            ]
            
            found_dir = None
            for alt_dir in alternative_dirs:
                if os.path.exists(alt_dir):
                    logger.log(f"Found alternative directory: {alt_dir}")
                    model_dir = alt_dir
                    found_dir = True
                    break
            
            if not found_dir:
                # Fall back to default
                logger.log("Using default model path")
                test_model(model_path="model_episode_110000.pth", episodes=6, opponent_type="aggressive")
                exit()
        
        # List all files in the directory
        all_files = os.listdir(model_dir)
        logger.log(f"All files in directory ({len(all_files)} files):")
        for file in all_files:
            logger.log(f"  - {file}")
        
        # First check for latest_model.pth
        latest_model_path = os.path.join(model_dir, "latest_model.pth")
        if os.path.exists(latest_model_path):
            logger.log(f"Found latest_model.pth - using this as priority")
            # Test against each opponent type
            logger.log("Testing against standard self-play opponent:")
            test_model(model_path=latest_model_path, episodes=2, opponent_type="self")
            logger.log("\nTesting against aggressive opponent:")
            test_model(model_path=latest_model_path, episodes=2, opponent_type="aggressive")
            logger.log("\nTesting against deceptive opponent:")
            test_model(model_path=latest_model_path, episodes=2, opponent_type="deceptive")
        else:
            # Filter for model files
            model_files = [f for f in all_files if f.startswith("model_episode_") and f.endswith(".pth")]
            
            if not model_files:
                logger.log(f"No model files found in {model_dir}")
                # Fall back to default
                logger.log("Using default model path")
                # Test against each opponent type
                logger.log("Testing against standard self-play opponent:")
                test_model(model_path="model_episode_110000.pth", episodes=2, opponent_type="self")
                logger.log("\nTesting against aggressive opponent:")
                test_model(model_path="model_episode_110000.pth", episodes=2, opponent_type="aggressive")
                logger.log("\nTesting against deceptive opponent:")
                test_model(model_path="model_episode_110000.pth", episodes=2, opponent_type="deceptive")
            else:
                # Sort by episode number to find the latest
                model_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
                logger.log(f"Found {len(model_files)} model files:")
                for i, model in enumerate(model_files):
                    logger.log(f"  {i+1}. {model} (Episode {model.split('_')[2].split('.')[0]})")
                
                latest_model = model_files[0]
                model_path = os.path.join(model_dir, latest_model)
                logger.log(f"Using latest model: {model_path} (Episode {latest_model.split('_')[2].split('.')[0]})")
                
                # Test against each opponent type
                logger.log("Testing against standard self-play opponent:")
                test_model(model_path=model_path, episodes=2, opponent_type="self")
                logger.log("\nTesting against aggressive opponent:")
                test_model(model_path=model_path, episodes=2, opponent_type="aggressive")
                logger.log("\nTesting against deceptive opponent:")
                test_model(model_path=model_path, episodes=2, opponent_type="deceptive")
    except Exception as e:
        logger.log(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure to close the log file
        logger.close()