# environment.py
import rlcard
import numpy as np
from collections import defaultdict
import os
import torch

class PokerEnv:
    def __init__(self):
        self.env = rlcard.make('no-limit-holdem', config={'num_players': 2})
        
        # Expand to more realistic NLHE actions
        self.num_actions = 7  # Fold, Call, Raise 1/3 pot, Raise 1/2 pot, Raise pot, Check, All-in
        
        self.starting_stack = 100  # Starting chips for each player
        self.player_stacks = [self.starting_stack, self.starting_stack]  # Track player stacks
        
        # Load or build equity table
        self.preflop_equity_table = self._load_or_build_preflop_equity_table()
        self.hand_strength_cache = {}  # Memoization cache
        
        # Opponent modeling
        self.opponent_stats = {
            'vpip': 0.0,  # Voluntary Put In Pot
            'pfr': 0.0,   # Pre-flop Raise
            'hands': 0,
            'fold_to_bet': 0.0,
            'bets_seen': 0
        }

    def _load_or_build_preflop_equity_table(self, filepath="preflop_equity_table.pth", force_rebuild=False):
        """
        Load preflop equity table from file if exists and is compatible, otherwise build and save it.
        """
        TABLE_VERSION = "1.0"  # Add versioning for future compatibility
        if os.path.exists(filepath) and not force_rebuild:
            try:
                data = torch.load(filepath)
                if isinstance(data, dict) and data.get("version") == TABLE_VERSION:
                    print(f"Loaded preflop equity table (v{TABLE_VERSION}) from {filepath}")
                    return data["equity_table"]
                else:
                    print(f"Version mismatch or invalid data in {filepath}. Rebuilding...")
            except Exception as e:
                print(f"Error loading equity table: {e}. Rebuilding...")
        else:
            print(f"{'Rebuilding' if force_rebuild else 'No'} preflop equity table found. Building a new one...")

        equity_table = self._build_preflop_equity_table()
        try:
            torch.save({"version": TABLE_VERSION, "equity_table": equity_table}, filepath)
            print(f"Saved preflop equity table (v{TABLE_VERSION}) to {filepath}")
        except Exception as e:
            print(f"Error saving equity table: {e}")
        return equity_table

    def _build_preflop_equity_table(self):
        """
        Build a preflop equity table for all 169 unique hands.
        These are simplified but more realistic approximations based on common poker equity data.
        """
        equity_table = {}
        suits = ['s', 'h', 'd', 'c']
        rank_symbols = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        # Pre-computed approximate equities vs. a random hand (based on typical poker data)
        # Pairs: 22 (~50%) to AA (~85%)
        pair_equities = np.linspace(0.50, 0.85, 13)  # 13 pairs
        for rank, equity in enumerate(pair_equities):
            rank_symbol = rank_symbols[rank]
            pair_key = tuple(sorted([f"{rank_symbol}s", f"{rank_symbol}h"]))
            equity_table[pair_key] = float(equity)  # Make sure it's a Python float for compatibility

        # Non-pairs (suited and offsuit)
        for high_rank in range(12, -1, -1):  # A (12) to 2 (0)
            high_symbol = rank_symbols[high_rank]
            for low_rank in range(high_rank - 1, -1, -1):
                low_symbol = rank_symbols[low_rank]
                
                # Suited hands (e.g., AKs, AQs)
                suited_key = tuple(sorted([f"{high_symbol}s", f"{low_symbol}s"]))
                # Base equity: higher card dominates, adjusted for connectedness
                base_equity = 0.35 + (high_rank / 36)  # ~0.35 to ~0.69
                gap = high_rank - low_rank
                suited_bonus = 0.08 if gap <= 4 else 0  # Connectedness bonus
                suited_equity = base_equity + suited_bonus
                equity_table[suited_key] = float(min(0.80, max(0.35, suited_equity)))
                
                # Offsuit hands (e.g., AKo, AQo)
                offsuit_key = tuple(sorted([f"{high_symbol}s", f"{low_symbol}h"]))
                offsuit_equity = suited_equity - 0.06  # Offsuit penalty
                equity_table[offsuit_key] = float(min(0.75, max(0.30, offsuit_equity)))

        return equity_table

    def reset(self):
        obs, player_id = self.env.reset()
        self.player_id = player_id
        self.street = 0  # Track street (0=preflop, 1=flop, 2=turn, 3=river)
        
        # Store current observation for legal action tracking
        self.current_obs = obs
        
        # Reset player stacks at the beginning of each hand
        self.player_stacks = [self.starting_stack, self.starting_stack]
        self.initial_stack = self.player_stacks[self.player_id]
        
        # Track chips committed to the pot
        self.chips_committed = 0
        
        # Determine if player is in big blind
        self.is_big_blind = self._is_big_blind()
        
        # Store the initial observation for EV calculations
        self.initial_obs = obs
        
        return self._get_obs(obs)

    def step(self, action):
        # Cache game state access
        game = self.env.game
        next_obs, player_id = self.env.step(self._translate_action(action))
        
        # Store current observation for legal action tracking
        self.current_obs = next_obs
        
        done = self.env.is_over()
        
        # Get game state values only once
        pot_size = game.dealer.pot
        player = game.players[self.player_id]
        opponent_id = 1 - self.player_id
        opponent = game.players[opponent_id]
        
        # Update chips committed
        if action in [1, 2, 3, 4, 6]:  # Any action that commits chips
            self.chips_committed = player.in_chips
        
        # Update opponent modeling stats
        if self.street == 0:
            if opponent.in_chips > 0:
                # Opponent has voluntarily put money in pot
                self.opponent_stats['vpip'] = ((self.opponent_stats['vpip'] * self.opponent_stats['hands']) + 1) / (self.opponent_stats['hands'] + 1)
            
            if opponent.in_chips > 2:  # More than BB - opponent raised
                self.opponent_stats['pfr'] = ((self.opponent_stats['pfr'] * self.opponent_stats['hands']) + 1) / (self.opponent_stats['hands'] + 1)
        
        # Track opponent fold behavior
        if action in [2, 3, 4, 6] and opponent.status == 'folded':  # We bet/raised and opponent folded
            self.opponent_stats['fold_to_bet'] = ((self.opponent_stats['fold_to_bet'] * self.opponent_stats['bets_seen']) + 1) / (self.opponent_stats['bets_seen'] + 1)
            self.opponent_stats['bets_seen'] += 1
        
        # Calculate equity and pot odds once
        equity = self._calculate_equity(next_obs['obs'])
        pot_odds = self._calculate_pot_odds(next_obs['obs'])
        
        reward_scaling_factor = 0.1  # Parameterize for tuning
        reward = 0
        if done:
            raw_reward = self.env.get_payoffs()[self.player_id]
            # Scale reward based on stack size and pot odds
            reward = raw_reward / self.starting_stack
            
            # Add GTO-like considerations
            pot_ratio = pot_size / (self.starting_stack * 2)
            
            if raw_reward > 0:
                reward += reward_scaling_factor * pot_ratio
                
                # Check if hand ended due to fold
                fold_detected = any(p.status == 'folded' for p in game.players)
                
                # Reward exploiting opponent folds more intelligently
                if fold_detected:
                    # Higher reward for successful bluffs (lower equity hands)
                    reward += reward_scaling_factor * 2 * (1 - equity) if equity < 0.3 else reward_scaling_factor / 2
                else:
                    # Reward for value betting strong hands that get called/shown down
                    reward += reward_scaling_factor * equity if equity > 0.6 else 0
            else:
                # Loss penalties - less severe for small pot losses, more severe for big ones
                loss_ratio = self.chips_committed / self.starting_stack
                
                # Smaller penalty for folding (minimizing losses)
                if action == 0:  # Fold
                    reward = -reward_scaling_factor * loss_ratio
                else:
                    # Calculate EV loss for non-fold actions
                    ev_loss = self._calculate_ev_loss(next_obs['obs'], action, equity, pot_odds)
                    reward = -loss_ratio - (reward_scaling_factor * ev_loss)
        else:
            # Intermediate rewards based on more accurate EV calculations
            ev_change = self._calculate_ev_change(next_obs['obs'], action, equity, pot_odds)
            reward = reward_scaling_factor * ev_change  # Smaller immediate rewards
            
            if self.street == 0:  # Preflop specific rewards
                if action == 0:  # Folding
                    if equity < pot_odds:
                        reward += 0.05  # Good fold with weak hand
                    elif equity > 0.5:
                        reward -= 0.1 * equity  # Bad fold with strong hand
                    
                elif action in [2, 3, 4]:  # Raising
                    # Reward raising with strong hands, penalize with weak
                    if equity > 0.6:
                        reward += 0.05
                    elif equity < 0.3:
                        reward -= 0.05
                    
                elif action == 1:  # Calling
                    # Getting the right price to call
                    if equity >= pot_odds - 0.05:
                        reward += 0.03
                    else:
                        reward -= 0.03
                        
                # Small bonus for checking in BB when appropriate
                if self.is_big_blind and action == 5:  # Check
                    reward += 0.02
            else:
                # Post-flop decisions
                implied_odds = self._calculate_implied_odds(next_obs['obs'])
                min_odds = min(pot_odds, implied_odds)
                
                if action == 0:  # Folding
                    # Mathematically correct folds
                    if equity < pot_odds:
                        reward += 0.04
                    elif equity > pot_odds + 0.1:
                        reward -= 0.06  # Penalty for folding when ahead
                
                elif action in [2, 3, 4]:  # Various raise sizes
                    # Value betting vs bluffing considerations
                    if equity > pot_odds + 0.2:
                        reward += 0.06  # Good value bet
                    elif equity < pot_odds - 0.1 and self.opponent_stats['fold_to_bet'] > 0.5:
                        reward += 0.04  # Good bluff against folding opponent
                    elif equity < pot_odds - 0.15:
                        reward -= 0.03  # Poor bluff against calling station
                
                elif action == 1:  # Calling
                    # Calling with correct odds (including implied)
                    if equity >= min_odds - 0.05:
                        reward += 0.03
                    else:
                        reward -= 0.03
                
                elif action == 5:  # Checking
                    # Check-calling with medium strength hands
                    if 0.3 < equity < 0.6:
                        reward += 0.02
                    elif equity > 0.7:
                        reward -= 0.02  # Should bet for value
            
            # Progressive reward for advancing streets with good equity
            new_street = self._get_street(next_obs['obs'])
            if new_street > self.street:
                # Bigger bonus for seeing flop
                if self.street == 0 and new_street == 1:
                    reward += 0.04
                else:
                    # Smaller bonuses for later streets, scaled by equity
                    reward += 0.02 + (0.01 * equity)
                
                self.street = new_street
                
                # Update opponent stats at end of hand
                if done:
                    self.opponent_stats['hands'] += 1
        
        next_state = self._get_obs(next_obs) if not done else (None, [])
        return next_state, reward, done

    def _get_obs(self, obs):
        full_obs = obs['obs']  # shape (54,)
        # Get legal actions directly from the observation
        legal_actions = list(self._get_legal_actions(obs).keys())
        return full_obs, legal_actions

    def _rank_to_symbol(self, rank):
        """Convert numeric rank to symbol"""
        symbols = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return symbols[rank]

    def _suit_to_symbol(self, suit):
        """Convert numeric suit to symbol"""
        return ['s', 'h', 'd', 'c'][suit]

    def _estimate_hand_strength(self, obs):
        """Vectorized hand strength calculation using NumPy"""
        # Use cache if available
        obs_key = tuple(obs[:52])
        if obs_key in self.hand_strength_cache:
            return self.hand_strength_cache[obs_key]
        
        # Vectorized calculation
        card_obs = obs[:52]
        card_indices = np.where(card_obs == 1)[0]
        
        # Early return for insufficient cards
        if len(card_indices) < 2:
            return 0.0
        
        # Extract ranks and suits with vectorized operations
        ranks = card_indices // 4
        suits = card_indices % 4
        
        if self.street == 0:  # Preflop
            # Check for pairs (look for duplicates in ranks)
            unique_ranks, rank_counts = np.unique(ranks, return_counts=True)
            pairs = np.sum(rank_counts >= 2)
            
            if pairs > 0:
                # Find the pair rank
                pair_rank = unique_ranks[rank_counts >= 2][0]
                # Normalize to 0-1 range
                return 0.5 + (pair_rank / 26)
            
            # Check for suited cards
            unique_suits, suit_counts = np.unique(suits, return_counts=True)
            suited = np.any(suit_counts >= 2)
            
            # Find highest and connected cards
            high_cards = np.sum(ranks >= 8)
            highest_rank = np.max(ranks)
            
            # Check for connected cards
            sorted_ranks = np.sort(ranks)
            gaps = np.diff(sorted_ranks)
            connected = np.any(gaps <= 3)
            
            # Base strength calculation
            base_strength = 0.2 + (highest_rank / 52)
            
            # Apply modifiers
            if suited:
                base_strength += 0.1
            if connected:
                base_strength += 0.08
            
            # Bonus for high cards
            base_strength += 0.05 * high_cards
            
            # Penalty for disconnected low cards
            if highest_rank < 8 and np.min(gaps) > 4 and not suited:
                base_strength *= 0.7
            
            result = np.clip(base_strength, 0.0, 1.0)
        else:
            # Postflop: Basic hand evaluation
            rank_counts = np.bincount(ranks, minlength=13)
            suit_counts = np.bincount(suits, minlength=4)
            
            # Check for pairs, trips, quads
            pairs = np.sum(rank_counts == 2)
            trips = np.sum(rank_counts == 3)
            quads = np.sum(rank_counts == 4)
            
            # Check for flush potential
            flush_potential = np.max(suit_counts) >= 2
            
            # Check for straight potential
            sorted_ranks = np.unique(ranks)
            straight_potential = False
            for i in range(len(sorted_ranks) - 4):
                if sorted_ranks[i + 4] - sorted_ranks[i] == 4:
                    straight_potential = True
                    break

            strength = (0.2 * pairs) + (0.4 * trips) + (0.8 * quads) + \
                       (0.15 if flush_potential else 0) + (0.15 if straight_potential else 0)
            
            # Adjust for high cards and community cards
            high_cards = np.sum(ranks >= 8)
            game = self.env.game
            strength += 0.05 * high_cards + 0.1 * (len(game.public_cards) / 5)
            
            result = np.clip(strength, 0.0, 1.0)
        
        # Cache the result
        self.hand_strength_cache[obs_key] = result
        return result

    def _calculate_pot_odds(self, obs):
        """Calculate pot odds with efficiency improvements"""
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
        # Use game state directly for efficiency
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

    def _get_legal_actions(self, obs=None):
        """
        Map RLCard legal actions to our expanded action space
        Modified to get legal actions from observation
        """
        if obs is None:
            legal = {0: 0, 1: 0}  # Default to fold and call
        else:
            legal = obs['legal_actions']

        game = self.env.game
        pot_size = game.dealer.pot
        stack = self.player_stacks[self.player_id]
        opponent_id = 1 - self.player_id
        to_call = max(0, game.players[opponent_id].in_chips - game.players[self.player_id].in_chips)

        action_map = {0: 0}  # Fold always available
        if 1 in legal and to_call <= stack:
            action_map[1] = to_call  # Call
        if 3 in legal and to_call == 0:
            action_map[5] = 0  # Check

        if 2 in legal and stack > to_call:
            min_raise = max(2, to_call + 2)  # Minimum raise size (e.g., big blind = 2)
            action_map[2] = min(stack, max(min_raise, int(pot_size * 0.33)))
            action_map[3] = min(stack, max(min_raise, int(pot_size * 0.5)))
            action_map[4] = min(stack, max(min_raise, int(pot_size)))
            action_map[6] = stack  # All-in

        return action_map

    def _translate_action(self, action):
        """Translate our expanded action space to RLCard's actions"""
        # Get legal actions from current observation
        legal = self.current_obs['legal_actions'] if hasattr(self, 'current_obs') else {}
        
        if action == 0:  # Fold
            return 0
        elif action == 1:  # Call
            return 1
        elif action == 5:  # Check
            return 3 if 3 in legal else 1  # Check if possible, otherwise call
        elif action == 6:  # All-in
            return 2  # Use RLCard's raise
        elif action in [2, 3, 4]:  # Various raise sizes
            return 2  # All map to RLCard's raise
        
        # Default fallback
        return 0

    def get_legal_actions(self):
        """Returns the list of legal actions"""
        if hasattr(self, 'current_obs') and self.current_obs:
            return list(self._get_legal_actions(self.current_obs).keys())
        else:
            # Fallback: assume basic actions are available based on game state
            game = self.env.game
            to_call = max(0, game.players[1 - self.player_id].in_chips - game.players[self.player_id].in_chips)
            actions = [0]  # Fold always available
            if to_call <= self.player_stacks[self.player_id]:
                actions.append(1)  # Call
            if to_call == 0:
                actions.append(5)  # Check
            if self.player_stacks[self.player_id] > to_call:
                actions.extend([2, 3, 4, 6])  # Raises and all-in
            return actions

    def _is_big_blind(self):
        """Determine if the player is in the big blind position"""
        game = self.env.game
        # In a 2-player game, player 1 is typically the big blind in the first round
        return self.player_id == 1

    def _calculate_equity(self, obs):
        """
        equity calculation using lookup tables for preflop
        and simplified Monte Carlo for postflop
        """
        if self.street == 0:
            # Use pre-computed lookup table for preflop
            hole_cards = self._get_hole_cards(obs)
            if len(hole_cards) < 2:
                return 0.0
                
            # Convert to card notation (e.g., 'As', 'Kh')
            card_strings = []
            for rank, suit in hole_cards:
                rank_symbol = self._rank_to_symbol(rank)
                suit_symbol = self._suit_to_symbol(suit)
                card_strings.append(f"{rank_symbol}{suit_symbol}")
            
            # Create lookup key
            card_key = tuple(sorted(card_strings))
            
            # Get equity from table or estimate if not found
            return self.preflop_equity_table.get(card_key, 0.5)
        else:
            # For postflop, use a simplified model since true Monte Carlo
            # would be computationally expensive
            hand_strength = self._estimate_hand_strength(obs)
            
            # Adjust hand strength based on community cards
            game = self.env.game
            num_community = len(game.public_cards)
            
            # As more community cards are revealed, hand strength becomes more accurate
            confidence_factor = 0.6 + (num_community * 0.1)  # 0.9 by river
            
            # Scale hand strength toward true equity and adjust for opponent tendencies
            equity = hand_strength * confidence_factor
            
            # Adjust equity based on opponent stats
            if self.opponent_stats['hands'] > 10:  # Only use once we have data
                if self.opponent_stats['vpip'] < 0.3:  # Tight opponent
                    # They likely have stronger hands when they play
                    equity *= 0.9
                elif self.opponent_stats['vpip'] > 0.6:  # Loose opponent
                    # They likely have weaker hands when they play
                    equity *= 1.1
            
            return min(1.0, max(0.0, equity))

    def _get_hole_cards(self, obs):
        """Extract hole cards from observation"""
        card_obs = obs[:52]
        hole_cards = []
        for i in range(52):
            if card_obs[i] == 1:
                card_rank = i // 4
                card_suit = i % 4
                hole_cards.append((card_rank, card_suit))
        return hole_cards

    def _get_board_cards(self, obs):
        """Extract board cards from game state"""
        game = self.env.game
        board_cards = []
        for card in game.public_cards:
            # Convert RLCard's card format to our (rank, suit) format
            # This is a simplified version - actual conversion depends on RLCard's format
            rank = card.rank
            suit = card.suit
            board_cards.append((rank, suit))
        return board_cards

    def _calculate_implied_odds(self, obs):
        # Base implied odds on the current pot odds
        pot_odds = self._calculate_pot_odds(obs)
        equity = self._calculate_equity(obs)
        game = self.env.game
        
        # Factor in remaining streets
        remaining_streets = 3 - self.street  # 3 = river
        
        # If we're drawing, implied odds matter more
        drawing_hand = 0.25 < equity < 0.45
        
        # Modifier based on stack sizes relative to pot
        pot_size = game.dealer.pot
        effective_stack = min(self.player_stacks)
        stack_to_pot = effective_stack / max(1, pot_size)
        
        # Use opponent modeling to adjust implied odds
        opp_fold_tendency = self.opponent_stats['fold_to_bet'] if self.opponent_stats['bets_seen'] > 5 else 0.3
        
        # Base implied odds are better than pot odds if we're drawing
        # and have deep stacks relative to the pot
        if drawing_hand:
            # Reduce implied odds as we get closer to the river
            street_factor = 0.6 + (remaining_streets * 0.2)
            
            # Deep stacks improve implied odds
            stack_factor = min(1.5, 0.8 + (stack_to_pot * 0.7))
            
            # Adjust for opponent tendencies - loose-passive opponents improve implied odds
            opp_factor = 1.0 + (0.3 * (1 - opp_fold_tendency))
            
            return max(pot_odds - 0.1, pot_odds * street_factor * stack_factor * opp_factor)
        else:
            # Non-drawing hands don't benefit as much from implied odds
            return pot_odds + 0.05

    def _calculate_ev_loss(self, obs, action, equity, pot_odds):
        """
        Calculate the Expected Value loss for a given action
        """
        game = self.env.game
        pot_size = game.dealer.pot
        
        # Basic EV calculation
        if action == 0:  # Fold
            # EV loss from folding = equity * pot_size
            # (what we expect to win if we had perfect information)
            return equity * pot_size / self.starting_stack
        elif action in [1, 2, 3, 4, 6]:  # Actions that commit chips
            # For call/raise actions, EV loss is based on equity vs pot odds
            if equity < pot_odds:
                # We're calling with negative EV
                return (pot_odds - equity) * (pot_size / self.starting_stack)
            else:
                # We're making a +EV call, no loss
                return 0
        else:
            # Checking has no direct EV loss
            return 0

    def _calculate_ev_change(self, obs, action, equity, pot_odds):
        """
        Calculate the Expected Value change from our action
        """
        game = self.env.game
        pot_size = game.dealer.pot
        
        # Normalize by starting stack for reasonable reward scaling
        norm_pot = pot_size / self.starting_stack
        
        if action == 0:  # Fold
            # Folding when behind is good, folding when ahead is bad
            ev_change = -equity * norm_pot
            return 1.0 if equity < pot_odds else ev_change
            
        elif action in [2, 3, 4]:  # Raising
            # EV of raising = fold equity + continue equity
            fold_equity = self.opponent_stats['fold_to_bet'] if self.opponent_stats['bets_seen'] > 5 else 0.3
            
            # When we raise, we want high fold equity when our hand is weak
            # and high continue equity when our hand is strong
            bluff_ev = fold_equity * norm_pot
            value_ev = (1 - fold_equity) * (2 * equity - 1) * norm_pot
            
            return max(bluff_ev, value_ev) * 2  # Scale up to make raises more impactful
            
        elif action == 1:  # Calling
            # EV of calling = equity - pot_odds
            return (equity - pot_odds) * 5  # Scale for more meaningful rewards
            
        elif action == 5:  # Checking
            # Checking is usually neutral, slightly positive with medium hands
            pot_control_bonus = 0.5 if 0.3 < equity < 0.6 else 0
            return pot_control_bonus
            
        elif action == 6:  # All-in
            # All-in decisions are high variance but should be based on equity
            ev = (2 * equity - 1) * 3  # Scale to make this a big decision
            return ev
            
        return 0  # Default