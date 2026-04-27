"""Play MCTS agent vs RandomPlayer (professor-style $1000 / SB $10 settings)."""

import sys

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

if __name__ == "__main__":
  config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=10)
  config.register_player(name="mcts", algorithm=MCTSPokerPlayer(time_budget=0.38))
  config.register_player(name="rand", algorithm=RandomPlayer())
  start_poker(config, verbose=1)
