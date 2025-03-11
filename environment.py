# environment.py
import rlcard
import numpy as np

class PokerEnv:
    def __init__(self):
        self.env = rlcard.make('no-limit-holdem', config={'num_players': 2})
        self.num_actions = 5  # Fold (0), Call (1), Raise small (2), Check (3), Raise large (4)
        self.starting_stack = 100  # Starting chips for each player
        self.player_stacks = [self.starting_stack, self.starting_stack]  # Track player stacks

    def reset(self):
        obs, player_id = self.env.reset()
        self.player_id = player_id
        self.street = 0  # Track street (0=preflop, 1=flop, 2=turn, 3=river)
        
        # Reset player stacks at the beginning of each hand
        self.player_stacks = [self.starting_stack, self.starting_stack]
        self.initial_stack = self.player_stacks[self.player_id]
        
        # Track chips committed to the pot
        self.chips_committed = 0
        
        return self._get_obs(obs)

    def step(self, action):
        next_obs, player_id = self.env.step(action)
        done = self.env.is_over()
        
        # Track chips committed to pot by our agent
        game = self.env.game
        if action in [1, 2, 4]:  # Call or Raise actions
            current_in_chips = game.players[self.player_id].in_chips
            self.chips_committed = current_in_chips
        
        reward = 0
        if done:
            # Calculate profit-based reward
            payoffs = self.env.get_payoffs()
            raw_reward = payoffs[self.player_id]
            
            # Scale reward based on profit relative to starting stack
            if raw_reward > 0:
                # Winning is good, but winning more is better
                # Normalize by starting stack to keep rewards in reasonable range
                reward = raw_reward / self.starting_stack
                
                # Bonus for winning a large pot
                pot_size = game.dealer.pot
                if pot_size > self.starting_stack * 0.5:
                    reward *= 1.2  # 20% bonus for winning large pots
                
                # Successful bluffs carrot
                hand_strength = self._estimate_hand_strength(next_obs['obs'])
                if hand_strength < 0.3 and raw_reward > 0:
                    reward *= 1.3  # 100% bonus for successful bluffs (increased from 30%)
                    print(f"Successful bluff! Hand strength: {hand_strength:.2f}, Reward: {reward:.2f}")
                elif hand_strength < 0.4 and raw_reward > 0:
                    reward *= 1.1  # 50% bonus for semi-bluffs
                    
            else:
                # Losing is bad, but minimize losses
                # If we folded early and lost little, that's better than losing everything
                chips_lost = self.chips_committed
                max_possible_loss = self.starting_stack
                
                # Calculate how much we saved by folding (if we did)
                if action == 0 and chips_lost < max_possible_loss:
                    saved_ratio = 1 - (chips_lost / max_possible_loss)
                    # Small negative reward, but not as bad as losing everything
                    reward = -0.2 + (saved_ratio * 0.15)  # Between -0.2 and -0.05
                else:
                    # Lost by showdown or other means
                    reward = raw_reward / self.starting_stack  # Normalized negative reward
        
        elif action == 0:  # Folding during the hand
            # Penalize folding with a strong hand
            hand_strength = self._estimate_hand_strength(next_obs['obs'])
            if hand_strength > 0.5:  # Strong hand
                reward = -0.3  # Significant penalty for folding a strong hand
            elif hand_strength > 0.3:  # Decent hand
                reward = -0.1  # Smaller penalty for folding a decent hand
            else:
                # Small positive reward for folding a weak hand when not forced to
                pot_odds = self._calculate_pot_odds(next_obs['obs'])
                if pot_odds < hand_strength:
                    reward = 0.05  # Small reward for making a mathematically correct fold
        
        elif action in [2, 4]:  # Raising actions - potential bluffs
            hand_strength = self._estimate_hand_strength(next_obs['obs'])
            pot_size = game.dealer.pot
            
            # ENHANCED: Small immediate reward for bluffing with weak hands
            # This encourages the agent to try bluffing more often
            if hand_strength < 0.3:
                # Small positive reward for raising with weak hands
                # The size depends on the pot - bigger bluffs on bigger pots
                bluff_reward = 0.02 * (pot_size / self.starting_stack)
                reward += bluff_reward
                
            # ENHANCED: Reward for semi-bluffing with drawing hands
            elif 0.3 <= hand_strength <= 0.5:
                # Smaller reward for semi-bluffing
                semi_bluff_reward = 0.01 * (pot_size / self.starting_stack)
                reward += semi_bluff_reward
        
        else:
            # Reward for actions during the hand based on pot odds and hand strength
            hand_strength = self._estimate_hand_strength(next_obs['obs'])
            pot_odds = self._calculate_pot_odds(next_obs['obs'])
            
            # Reward aggressive play with strong hands
            if hand_strength > 0.7 and action in [2, 4]:  # Strong hand and raising
                reward = 0.05  # Small positive reward
            
            # Reward checking with medium-strength hands
            elif 0.3 < hand_strength < 0.6 and action == 3:
                reward = 0.02  # Tiny positive reward
            
            # Small reward for progressing to new streets
            new_street = self._get_street(next_obs['obs'])
            if new_street > self.street:
                reward = 0.01 * new_street  # Increasing reward for later streets
                self.street = new_street
        
        next_state = self._get_obs(next_obs) if not done else (None, [])
        return next_state, reward, done

    def _get_obs(self, obs):
        full_obs = obs['obs']  # shape (54,)
        legal_actions = list(obs['legal_actions'].keys())
        return full_obs, legal_actions

    def _estimate_hand_strength(self, obs):
        """Estimate the strength of the current hand (0-1 scale)"""
        # This is a simplified version - you could implement a more sophisticated hand evaluator
        card_obs = obs[:52]  # First 52 elements represent cards
        
        # Count high cards (face cards and aces) in hand
        high_cards = sum(card_obs[36:52])  # Indices for face cards and aces
        
        # Check for pairs or better
        pairs = 0
        for i in range(0, 52, 4):  # Check each rank (4 cards per rank)
            if sum(card_obs[i:i+4]) >= 2:
                pairs += 1
        
        # Check for suited cards
        suited = False
        for suit in range(4):
            suit_cards = sum(card_obs[suit::4])  # Count cards of this suit
            if suit_cards >= 2:
                suited = True
        
        # Combine factors for a rough hand strength estimate
        strength = (0.1 * high_cards) + (0.2 * pairs) + (0.1 if suited else 0)
        
        # Add some weight for community cards that might help
        community_cards = obs[52:]  # Last elements might indicate community cards
        if sum(community_cards) > 0:
            strength += 0.1  # Bonus for having community cards that might help
        
        return min(1.0, max(0.0, strength))  # Ensure value is between 0 and 1

    def _calculate_pot_odds(self, obs):
        """Calculate pot odds (call amount / potential win)"""
        game = self.env.game
        pot_size = game.dealer.pot
        
        # If we're player 0, opponent is 1 and vice versa
        opponent_id = 1 - self.player_id
        
        # Calculate how much we need to call
        our_chips = game.players[self.player_id].in_chips
        opponent_chips = game.players[opponent_id].in_chips
        call_amount = max(0, opponent_chips - our_chips)
        
        if pot_size + call_amount == 0:
            return 0  # Avoid division by zero
            
        # Pot odds = call amount / (pot size + call amount)
        pot_odds = call_amount / (pot_size + call_amount)
        return pot_odds

    def _get_street(self, obs):
        # Count community cards to determine street
        game = self.env.game
        num_community_cards = len(game.public_cards)
        
        if num_community_cards == 0:
            return 0  # Preflop
        elif num_community_cards == 3:
            return 1  # Flop
        elif num_community_cards == 4:
            return 2  # Turn
        elif num_community_cards == 5:
            return 3  # River
        return self.street  # Default to current street

    def get_legal_actions(self):
        return list(self.env.get_legal_actions().keys())