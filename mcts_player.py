"""
Monte Carlo Tree Search poker agent (determinization / imperfect-information MCTS).

Uses root sampling: each iteration draws a full world (opponent hole cards + board runout),
evaluates chip-style EV for every legal action at showdown, and aggregates means.
Optional JSON bias (from offline self-play) shifts Q-values.

Course note: put your tournament logic in declare_action; keep helpers in this file or
import them — do not modify pypokerengine for submission.
"""

from __future__ import annotations

import json
import os
import random
import time

from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator

_CONFIG_NAME = "mcts_config.json"


def _load_optional_config(path: str | None) -> dict:
  if not path or not os.path.isfile(path):
    return {}
  try:
    with open(path, encoding="utf-8") as f:
      return json.load(f)
  except (json.JSONDecodeError, OSError):
    return {}


def _cards_from_str_list(cards_str: list[str]) -> list[Card]:
  return [Card.from_str(s) for s in cards_str]


def _remaining_deck_ids(known: list[Card]) -> list[int]:
  known_ids = {c.to_id() for c in known}
  return [i for i in range(1, 53) if i not in known_ids]


def _sample_world(
    hole_me: list[Card],
    community: list[Card],
    rng: random.Random,
) -> tuple[list[Card], list[Card]]:
  """Sample opponent hole cards and complete the board to five cards."""
  known = hole_me + community
  pool = _remaining_deck_ids(known)
  rng.shuffle(pool)
  opp_hole = [Card.from_id(pool[0]), Card.from_id(pool[1])]
  need = 5 - len(community)
  rest_ids = pool[2 : 2 + need]
  rest = [Card.from_id(cid) for cid in rest_ids]
  board5 = community + rest
  return opp_hole, board5


def _win_value(hole_me: list[Card], hole_opp: list[Card], board5: list[Card]) -> float:
  s_me = HandEvaluator.eval_hand(hole_me, board5)
  s_opp = HandEvaluator.eval_hand(hole_opp, board5)
  if s_me > s_opp:
    return 1.0
  if s_me < s_opp:
    return 0.0
  return 0.5


def _raise_add_amount(small_blind: int, street: str) -> int:
  # Matches engine: preflop/flop use 2*sb increment; turn/river use 4*sb
  if street in ("preflop", "flop"):
    return small_blind * 2
  return small_blind * 4


def _chip_ev_action(
    action: str,
    win: float,
    pot_main: float,
    small_blind: int,
    street: str,
) -> float:
  """
  Limit-style chip EV vs folding (fold EV = 0). Uses blind-sized units when exact
  call/raise amounts are not in valid_actions.
  """
  call_cost = float(max(small_blind * 2, 1))  # rough BB / amount to match
  if action == "fold":
    return 0.0
  if action == "call":
    return win * (pot_main + call_cost) - call_cost
  if action == "raise":
    add = float(_raise_add_amount(small_blind, street))
    raise_cost = call_cost + add
    return win * (pot_main + raise_cost) - raise_cost
  return 0.0


class MCTSPokerPlayer(BasePokerPlayer):
  """
  Root parallel MCTS with determinization (self-play / ISMCTS-style sampling).

  Parameters
  ----------
  time_budget : float
      Seconds for declare_action (stay below engine timeout, typically 0.5s).
  config_path : str | None
      Optional JSON with keys: action_bias (fold/call/raise floats), seed (int).
  """

  def __init__(self, time_budget: float = 0.38, config_path: str | None = None):
    super().__init__()
    self.time_budget = time_budget
    base = os.path.dirname(os.path.abspath(__file__))
    self.config_path = config_path or os.path.join(base, _CONFIG_NAME)
    self._cfg = _load_optional_config(self.config_path)
    self._bias = self._cfg.get("action_bias", {}) or {}
    seed = self._cfg.get("seed")
    self._rng = random.Random(seed)

  def declare_action(self, valid_actions, hole_card, round_state):
    t0 = time.perf_counter()
    deadline = t0 + self.time_budget

    hole_me = _cards_from_str_list(hole_card)
    community = _cards_from_str_list(round_state.get("community_card", []))
    pot_main = float(round_state["pot"]["main"]["amount"])
    sb = int(round_state["small_blind_amount"])
    street = round_state["street"]

    actions = [a["action"] for a in valid_actions]
    n = len(actions)
    visits = [0] * n
    value_sum = [0.0] * n

    rng = self._rng

    while time.perf_counter() < deadline:
      opp_hole, board5 = _sample_world(hole_me, community, rng)
      win = _win_value(hole_me, opp_hole, board5)
      for i, act in enumerate(actions):
        q = _chip_ev_action(act, win, pot_main, sb, street)
        b = float(self._bias.get(act, 0.0))
        visits[i] += 1
        value_sum[i] += q + b

    means = [value_sum[i] / max(visits[i], 1) for i in range(n)]
    best = max(range(n), key=lambda i: means[i])
    return actions[best]

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
  return MCTSPokerPlayer()
