import json
import random
import os
import numpy as np

STREETS = ["preflop", "flop", "turn", "river"]
MAX_RAISES = 4
ACTIONS = ["fold", "call", "raise"]   # indices 0, 1, 2
EQUITY_MAP = {1: 0.15, 2: 0.30, 3: 0.50, 4: 0.70, 5: 0.85}

def pot_bucket(pot, stacks):
    total = pot + sum(stacks)
    r = pot / (total + 1e-6)
    if r < 0.15: return 1
    if r < 0.40: return 2
    return 3

def n_actions(raise_count):
    return 3 if raise_count < MAX_RAISES else 2

def infoset_key(street, my_hand_bkt, pot_bkt, raise_count):
    return (street, my_hand_bkt, pot_bkt, min(raise_count, MAX_RAISES))

class InfoSet:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.ones(num_actions) / num_actions
        self.strategy_sum = np.zeros(num_actions)
        self.reach_prob = 0.0
        self.reach_prob_sum = 0.0

    def get_strategy(self):
        pos_regrets = np.maximum(self.regret_sum, 0.0)
        total = pos_regrets.sum()
        if total > 0:
            return pos_regrets / total
        return np.ones(self.num_actions) / self.num_actions

    def get_average_strategy(self):
        if self.reach_prob_sum > 0:
            avg = self.strategy_sum / self.reach_prob_sum
        else:
            avg = self.strategy_sum.copy()
        total = avg.sum()
        if total > 0:
            return avg / total
        return np.ones(self.num_actions) / self.num_actions

    def update(self):
        self.strategy_sum += self.reach_prob * self.strategy
        self.strategy = self.get_strategy()
        self.reach_prob_sum += self.reach_prob
        self.reach_prob = 0.0
        self.regret_sum = np.maximum(self.regret_sum, 0.0)

class GameState:
    def __init__(self, street_idx, hand_bkt, pot_bkt, raise_count,
                 pot, stack_p0, stack_p1, acting_player, folded_player=None):
        self.street_idx = street_idx
        self.hand_bkt = hand_bkt
        self.pot_bkt = pot_bkt
        self.raise_count = raise_count
        self.pot = pot
        self.stack = [stack_p0, stack_p1]
        self.acting_player = acting_player
        self.folded_player = folded_player

    def is_terminal(self):
        return self.street_idx > 3 or self.folded_player is not None

    def terminal_utility(self, perspective=0):
        if self.folded_player == perspective:
            return -self.pot
        if self.folded_player == 1 - perspective:
            return self.pot

        my_eq = np.clip(EQUITY_MAP[self.hand_bkt[perspective]] + np.random.normal(0, 0.03), 0, 1)
        opp_eq = np.clip(EQUITY_MAP[self.hand_bkt[1-perspective]] + np.random.normal(0, 0.03), 0, 1)

        win_prob = my_eq / (my_eq + opp_eq + 1e-6)
        return (2 * win_prob - 1) * self.pot

infosets = {}

def get_infoset(key, num_act):
    if key not in infosets:
        infosets[key] = InfoSet(num_act)
    return infosets[key]

def cfr(state, reach_p0, reach_p1, traverser):
    if state.is_terminal():
        return state.terminal_utility(perspective=traverser)

    p = state.acting_player
    street = STREETS[state.street_idx]
    num_act = n_actions(state.raise_count)
    key = infoset_key(street, state.hand_bkt[p], state.pot_bkt, state.raise_count)
    node = get_infoset(key, num_act)

    if p == traverser:
        node.reach_prob += reach_p0 if p == 0 else reach_p1

    strategy = node.strategy
    action_utils = np.zeros(num_act)

    if p == traverser:
        for i in range(num_act):
            ns = _apply_action(state, i)
            action_utils[i] = cfr(
                ns,
                reach_p0 * strategy[i] if p == 0 else reach_p0,
                reach_p1 * strategy[i] if p == 1 else reach_p1,
                traverser
            )
    else:
        i = np.random.choice(num_act, p=strategy)
        ns = _apply_action(state, i)
        action_utils[i] = cfr(
            ns,
            reach_p0 * strategy[i] if p == 0 else reach_p0,
            reach_p1 * strategy[i] if p == 1 else reach_p1,
            traverser
        )

    node_util = float(np.dot(action_utils, strategy))

    if p == traverser:
        opp_reach = reach_p1 if p == 0 else reach_p0
        node.regret_sum += opp_reach * (action_utils - node_util)

    return node_util

def _apply_action(state, action_idx):
    pot = state.pot
    stacks = list(state.stack)
    next_p = 1 - state.acting_player

    if action_idx == 0:
        return GameState(state.street_idx, state.hand_bkt, state.pot_bkt,
                         state.raise_count, pot, stacks[0], stacks[1],
                         next_p, folded_player=state.acting_player)

    elif action_idx == 1:
        amt = min(10, stacks[state.acting_player])
        stacks[state.acting_player] -= amt
        pot += amt
        new_pb = pot_bucket(pot, stacks)
        return GameState(state.street_idx + 1, state.hand_bkt, new_pb,
                         0, pot, stacks[0], stacks[1], 0)

    else:
        amt = min(20, stacks[state.acting_player])
        stacks[state.acting_player] -= amt
        pot += amt
        new_pb = pot_bucket(pot, stacks)
        return GameState(state.street_idx, state.hand_bkt, new_pb,
                         state.raise_count + 1, pot, stacks[0], stacks[1], next_p)

def train(iterations=100000):
    print(f"Running CFR+ for {iterations} iterations...")

    for i in range(iterations):
        hand_bkt = [random.randint(1, 5), random.randint(1, 5)]

        root = GameState(
            street_idx=0,
            hand_bkt=hand_bkt,
            pot_bkt=1,
            raise_count=0,
            pot=30,
            stack_p0=9980,
            stack_p1=9980,
            acting_player=0
        )

        traverser = i % 2
        cfr(root, 1.0, 1.0, traverser)

        for node in infosets.values():
            node.update()

        if (i + 1) % 20000 == 0:
            print(f"Iteration {i + 1}/{iterations} — {len(infosets)} infosets")

    strategy_table = {}
    for key, node in infosets.items():
        avg = node.get_average_strategy()
        num_act = node.num_actions

        str_key = f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"

        strategy_table[str_key] = {
            "fold": float(avg[0]),
            "call": float(avg[1]),
            "raise": float(avg[2]) if num_act == 3 else 0.0
        }

    out = os.path.join(os.path.dirname(__file__), "cfr_strategy.json")
    with open(out, "w") as f:
        json.dump(strategy_table, f, indent=2)

    print(f"\nCFR complete. {len(strategy_table)} infosets saved to {out}")

    return strategy_table

if __name__ == "__main__":
    train(iterations=100000)