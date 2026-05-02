from pypokerengine.players import BasePokerPlayer
import json
import os
import random

EQUITY_MAP = {1: 0.15, 2: 0.30, 3: 0.50, 4: 0.70, 5: 0.85}

class CFRPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.strategy = self._load_strategy()
        self.raise_count = 0

    def _load_strategy(self):
        path = os.path.join(os.path.dirname(__file__), "cfr_strategy.json")
        with open(path, "r") as f:
            return json.load(f)

    def _hand_bucket(self, hole_card):
        ranks = "23456789TJQKA"
        r1, r2 = ranks.index(hole_card[0][1]), ranks.index(hole_card[1][1])
        avg_rank = (r1 + r2) / 2
        if avg_rank < 3: return 1
        if avg_rank < 6: return 2
        if avg_rank < 9: return 3
        if avg_rank < 11: return 4
        return 5

    def _pot_bucket(self, round_state):
        pot = round_state["pot"]["main"]["amount"]
        stacks = sum(s["stack"] for s in round_state["seats"])
        r = pot / (pot + stacks + 1e-6)
        if r < 0.15: return 1
        if r < 0.40: return 2
        return 3

    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        hand_bkt = self._hand_bucket(hole_card)
        pot_bkt = self._pot_bucket(round_state)
        
        key = f"{street}|{hand_bkt}|{pot_bkt}|{min(self.raise_count, 4)}"
        probs = self.strategy.get(key, {"fold": 0.33, "call": 0.34, "raise": 0.33})

        r = random.random()
        if r < probs["fold"]:
            chosen = "fold"
        elif r < probs["fold"] + probs["call"]:
            chosen = "call"
        else:
            chosen = "raise"

        for va in valid_actions:
            if va["action"] == chosen:
                if chosen == "raise":
                    self.raise_count += 1
                return va["action"]
        
        return valid_actions[1]["action"]

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.raise_count = 0

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return CFRPlayer()
