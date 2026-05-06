import json
import os
import random

from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator

MAX_RAISES = 4
# Midpoints between EQUITY_MAP values {1:0.15, 2:0.30, 3:0.50, 4:0.70, 5:0.85}
_BUCKET_THRESHOLDS = [0.225, 0.40, 0.60, 0.775]


def _win_prob_to_bucket(wp: float) -> int:
    for bkt, thresh in enumerate(_BUCKET_THRESHOLDS, start=1):
        if wp < thresh:
            return bkt
    return 5


def _estimate_win_prob(hole_me: list, community: list, n_samples: int = 80) -> float:
    known_ids = {c.to_id() for c in hole_me + community}
    deck = [i for i in range(1, 53) if i not in known_ids]
    need = 5 - len(community)
    wins = 0.0
    for _ in range(n_samples):
        sample = random.sample(deck, 2 + need)
        opp_hole = [Card.from_id(sample[0]), Card.from_id(sample[1])]
        board5 = community + [Card.from_id(i) for i in sample[2:]]
        s_me = HandEvaluator.eval_hand(hole_me, board5)
        s_opp = HandEvaluator.eval_hand(opp_hole, board5)
        if s_me > s_opp:
            wins += 1.0
        elif s_me == s_opp:
            wins += 0.5
    return wins / n_samples


def _pot_bucket(pot: float, stacks: list) -> int:
    total = pot + sum(stacks)
    r = pot / (total + 1e-6)
    if r < 0.15:
        return 1
    if r < 0.40:
        return 2
    return 3


def _count_raises(action_histories: dict, street: str) -> int:
    history = action_histories.get(street, [])
    return sum(1 for a in history if a.get("action") == "RAISE")


class CustomPlayer(BasePokerPlayer):
    """Plays according to a pre-trained CFR strategy loaded from cfr_strategy.json."""

    def __init__(self, strategy_path: str = None, n_samples: int = 80):
        super().__init__()
        base = os.path.dirname(os.path.abspath(__file__))
        path = strategy_path or os.path.join(base, "cfr_strategy.json")
        with open(path, encoding="utf-8") as f:
            self._strategy: dict = json.load(f)
        self.n_samples = n_samples

    def declare_action(self, valid_actions, hole_card, round_state):
        hole_me = [Card.from_str(s) for s in hole_card]
        community = [Card.from_str(s) for s in round_state.get("community_card", [])]

        street: str = round_state["street"]
        pot = float(round_state["pot"]["main"]["amount"])
        stacks = [s["stack"] for s in round_state["seats"]]

        action_histories = round_state.get("action_histories", {})
        raise_count = min(_count_raises(action_histories, street), MAX_RAISES)

        wp = _estimate_win_prob(hole_me, community, self.n_samples)
        hand_bkt = _win_prob_to_bucket(wp)
        pot_bkt = _pot_bucket(pot, stacks)

        key = f"{street}|{hand_bkt}|{pot_bkt}|{raise_count}"
        available = [a["action"] for a in valid_actions]

        if key in self._strategy:
            probs = self._strategy[key]
            weights = [max(probs.get(a, 0.0), 0.0) for a in available]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                return random.choices(available, weights=weights)[0]

        # Fallback: lean on hand strength
        if wp >= 0.60 and "raise" in available:
            return "raise"
        if wp >= 0.35 and "call" in available:
            return "call"
        return available[0]  # fold

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return CustomPlayer()
