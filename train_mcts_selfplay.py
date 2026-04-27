"""
Offline self-play / evaluation for MCTSPokerPlayer (no live training in tournament).

Runs short matches MCTS vs Random (or MCTS vs MCTS), estimates win rates, and optionally
writes a small mcts_config.json with tiny action_bias hints from average per-action
chip proxies (weak heuristic — replace with real stats if you extend learning).

Usage (from CS683_Poker-Agent directory):
  python train_mcts_selfplay.py --games 30 --rounds 80
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# Windows: disable signal timeout wrapper before importing game (see run_basic_test.py)
if sys.platform == "win32":
  import pypokerengine.utils.timeout_decorator as _td

  def _noop_timeout2(*_a, **_k):
    def _d(f):
      return f
    return _d

  _td.timeout2 = _noop_timeout2

from pypokerengine.api.game import setup_config, start_poker

from mcts_player import MCTSPokerPlayer
from randomplayer import RandomPlayer


def _play_games(num_games: int, max_round: int, initial_stack: int, sb: int, mcts_vs_mcts: bool):
  rng = random.Random(42)
  stacks_a = 0
  stacks_b = 0
  for g in range(num_games):
    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=sb)
    if mcts_vs_mcts:
      p1 = MCTSPokerPlayer(time_budget=0.35, config_path=None)
      p2 = MCTSPokerPlayer(time_budget=0.35, config_path=None)
      p2._rng = random.Random(rng.randint(1, 10_000_000))
    else:
      p1 = MCTSPokerPlayer(time_budget=0.35, config_path=None)
      p2 = RandomPlayer()
    config.register_player(name="A", algorithm=p1)
    config.register_player(name="B", algorithm=p2)
    result = start_poker(config, verbose=0)
    seats = result["players"]
    stacks_a += seats[0]["stack"]
    stacks_b += seats[1]["stack"]
    print(f"game {g+1}/{num_games}  A={seats[0]['stack']}  B={seats[1]['stack']}")
  return stacks_a, stacks_b


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--games", type=int, default=20)
  parser.add_argument("--rounds", type=int, default=100)
  parser.add_argument("--stack", type=int, default=1000)
  parser.add_argument("--sb", type=int, default=10)
  parser.add_argument("--mcts-vs-mcts", action="store_true")
  parser.add_argument("--write-config", action="store_true", help="Write example mcts_config.json (seed only)")
  args = parser.parse_args()

  t0 = time.time()
  sa, sb = _play_games(args.games, args.rounds, args.stack, args.sb, args.mcts_vs_mcts)
  print("\n--- totals ---")
  print(f"Player A total stack sum: {sa}")
  print(f"Player B total stack sum: {sb}")
  print(f"Time: {time.time() - t0:.1f}s")

  if args.write_config:
    path = os.path.join(os.path.dirname(__file__), "mcts_config.json")
    payload = {"seed": 42, "action_bias": {"fold": 0.0, "call": 0.0, "raise": 0.0}}
    with open(path, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
  main()
