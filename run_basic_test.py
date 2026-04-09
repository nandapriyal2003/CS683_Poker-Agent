"""
Quick smoke test: two simple bots play a short game with console trace.
Run from the CS683_Poker-Agent folder:  python run_basic_test.py

Course compliance (PDF §5.1): submit only your player’s declare_action() — do not submit
this script or rely on engine edits. On Windows, SIGALRM is missing; the block below
disables the signal-based action timeout for local runs only (not needed on Linux/WSL).
"""
import sys

if sys.platform == "win32":
  import pypokerengine.utils.timeout_decorator as _td

  def _noop_timeout2(*_args, **_kwargs):
    def _decorate(function):
      return function
    return _decorate

  _td.timeout2 = _noop_timeout2

from pypokerengine.api.game import setup_config, start_poker

from always_call_player import AlwaysCallPlayer
from always_fold_player import AlwaysFoldPlayer

if __name__ == "__main__":
  config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
  config.register_player(name="caller", algorithm=AlwaysCallPlayer())
  config.register_player(name="folder", algorithm=AlwaysFoldPlayer())

  game_result = start_poker(config, verbose=1)
  print("\n--- final stacks ---")
  for seat in game_result["players"]:
    print(seat["name"], seat["stack"])
