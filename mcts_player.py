from pypokerengine.players import BasePokerPlayer
import random
import math
import copy

class MCTSNode:
    """Node for MCTS"""
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []

    def ucb_score(self, total_visits, explore_const=math.sqrt(2)):
        """Calculate UCB with sqrt(2) as default tuning parameter"""
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + explore_const * math.sqrt(2 * math.log(total_visits) / self.visits)


class MCTSPokerPlayer(BasePokerPlayer):
    """Poker agent using MCTS"""

    def __init__(self, iterations=100, explore_const=math.sqrt(2), n_samples=5):
        super().__init__()
        self.iterations = iterations
        self.explore_const = explore_const
        self.n_samples = n_samples
        self.uuid = None


    def declare_action(self, valid_actions, hole_card, round_state):
        """Build game state"""
        # Extract pot amount
        pot_amount = 0
        if 'pot' in round_state:
            if isinstance(round_state['pot'], dict):
                pot_amount = round_state['pot'].get('main', 0)
            else:
                pot_amount = round_state['pot']
        
        # Get current bet amount
        current_bet = 0
        if 'current_bet' in round_state:
            if isinstance(round_state['current_bet'], dict):
                # Find maximum bet any player has made
                for uuid, bet in round_state['current_bet'].items():
                    if isinstance(bet, (int, float)) and bet > current_bet:
                        current_bet = bet
            elif isinstance(round_state['current_bet'], (int, float)):
                current_bet = round_state['current_bet']
        
        # Get my stack
        my_stack = self._get_my_stack(round_state)
        
        # Get opponent's stack
        opponent_stack = self._get_opponent_stack(round_state)
        
        # Create clean state with only integers
        game_state = {
            'hole_card': hole_card,
            'valid_actions': valid_actions,
            'street': round_state.get('street', 'preflop'),
            'pot': pot_amount,
            'current_bet': current_bet,
            'community_cards': round_state.get('community_card', []),
            'my_stack': my_stack,
            'opponent_stack': opponent_stack,
            'terminal': False
        }

        # Run MCTS, return best action
        return self._mcts_search(game_state, valid_actions)
        
    def _get_my_stack(self, round_state):
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                return seat['stack']
        return 0

    def _get_opponent_stack(self, round_state):
        for seat in round_state['seats']:
            if seat['uuid'] != self.uuid:
                return seat['stack']
        return 0

    def _mcts_search(self, game_state, valid_actions):
        root = MCTSNode(game_state)
        root.untried_actions = valid_actions.copy()

        for _ in range(self.iterations):
            node = root

            # Step 1: Selection
            while node.children and not self._is_terminal(node.state):
                if node.untried_actions:
                    break
                node = self._select_child(node)

            # Step 2: Expansion
            if node.untried_actions and not self._is_terminal(node.state):
                node = self._expand(node)

            # Step 3: Simulation
            value = self._simulate(node.state)

            # Step 4: Backpropagation
            while node:
                node.visits += 1
                node.value += value
                node = node.parent

        # Pick best action
        if not root.children:
            # Default to call or check
            for action in valid_actions:
                if action['action'] == 'call':
                    return 'call'
            return valid_actions[0]['action']

        best_child = max(root.children.values(), key=lambda c: c.visits)
        return best_child.action

    def _select_child(self, node):
        total_visits = node.visits
        return max(node.children.values(), key=lambda c: c.ucb_score(total_visits, self.explore_const))

    def _expand(self, node):
        action_info = node.untried_actions.pop()
        action = action_info['action']

        new_state = self._apply_action(node.state, action_info)
        new_valid_actions = self._get_valid_actions(new_state)

        child = MCTSNode(new_state, parent=node, action=action)
        child.untried_actions = new_valid_actions
        node.children[action] = child

        return child

    def _simulate(self, state):
        total_value = 0

        for _ in range(self.n_samples):
            sampled_hand = self._sample_opponent_hand(state)
            value = self._rollout(state, sampled_hand)
            total_value += value

        return total_value / self.n_samples

    def _sample_opponent_hand(self, state):
        """Simple sampling: return random cards"""
        known_cards = state['hole_card'] + state['community_cards']
        ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
        suits = ['H', 'D', 'C', 'S']

        available = []
        for rank in ranks:
            for suit in suits:
                card = rank + suit
                if card not in known_cards:
                    available.append(card)

        if len(available) >= 2:
            return random.sample(available, 2)
        return ['AH', 'AS']

    def _rollout(self, state, opponent_hand):
        """Simple heuristic rollout"""
        hand_strength = self._eval_hand_strength(state['hole_card'], state['community_cards'])

        if hand_strength > 0.7:
            return 1.0
        elif hand_strength < 0.3:
            return -0.5
        else:
            return 0.0

    def _eval_hand_strength(self, hole_cards, community_cards):
        """Simplified hand strength evaluation"""
        if not community_cards:
            # Pre-flop evaluation
            if hole_cards[0][0] == hole_cards[1][0]:
                return 0.8  # Pocket pair
            elif hole_cards[0][0] in ['A', 'K', 'Q', 'J'] or hole_cards[1][0] in ['A', 'K', 'Q', 'J']:
                return 0.6  # High card
            else:
                return 0.4  # Low cards
        else:
            # Simple random for demonstration
            return random.random()

    def _apply_action(self, state, action_info):
        """Apply an action and return new state"""
        # Create a deep copy
        new_state = copy.deepcopy(state)
        action = action_info['action']
        
        # Ensure pot is an integer (in case any dict slipped through)
        if isinstance(new_state['pot'], dict):
            new_state['pot'] = new_state['pot'].get('main', 0)
        
        if action == 'fold':
            new_state['terminal'] = True
            return new_state
            
        elif action == 'call':
            # Calculate call amount
            call_amount = new_state['current_bet']
            if call_amount > new_state['my_stack']:
                call_amount = new_state['my_stack']
            
            # Update stacks and pot
            new_state['my_stack'] = new_state['my_stack'] - call_amount
            new_state['pot'] = new_state['pot'] + call_amount
            
        elif action == 'raise':
            # Get raise amount from action_info
            raise_amount = action_info.get('amount', 0)
            
            # For raise, need to call first then add raise
            call_amount = new_state['current_bet']
            total_bet = call_amount + raise_amount
            
            if total_bet > new_state['my_stack']:
                total_bet = new_state['my_stack']
            
            # Update stacks and pot
            new_state['my_stack'] = new_state['my_stack'] - total_bet
            new_state['pot'] = new_state['pot'] + total_bet
            new_state['current_bet'] = total_bet
        
        return new_state
    
    def _get_valid_actions(self, state):
        """Generate valid actions based on current state"""
        actions = []
        
        # Always can fold
        actions.append({'action': 'fold'})
        
        # Call/check
        actions.append({'action': 'call'})
        
        # Raise if enough chips
        min_raise = max(state['current_bet'] * 2, 20) if state['current_bet'] > 0 else 20
        if state['my_stack'] > min_raise:
            # Add a few raise amounts
            actions.append({'action': 'raise', 'amount': min_raise})
            
            # Add half-pot raise
            half_pot = max(min_raise, state['pot'] // 2)
            if half_pot <= state['my_stack']:
                actions.append({'action': 'raise', 'amount': half_pot})
            
            # Add pot-sized raise
            pot_raise = max(min_raise, state['pot'])
            if pot_raise <= state['my_stack']:
                actions.append({'action': 'raise', 'amount': pot_raise})
            
            # All-in
            actions.append({'action': 'raise', 'amount': state['my_stack']})
        
        return actions
    
    def _is_terminal(self, state):
        return state.get('terminal', False) or state.get('street') == 'showdown'
    
    def receive_game_start_message(self, game_info):
        # Find this player's uuid
        for seat in game_info['seats']:
            if seat.get('name') == 'MCTSPokerPlayer':
                self.uuid = seat['uuid']
                break
        # If not found, take first seat
        if not self.uuid and game_info['seats']:
            self.uuid = game_info['seats'][0]['uuid']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return MCTSPokerPlayer(iterations=500)
