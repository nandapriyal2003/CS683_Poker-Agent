from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from mcts_player import MCTSPokerPlayer

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=10)

# Pick matchup: change to test different matchups
matchup = "mcts_vs_random"

if matchup == "mcts_vs_random":
    config.register_player(name="MCTS_Agent", algorithm=MCTSPokerPlayer())
    config.register_player(name="RandomBot", algorithm=RandomPlayer())
elif matchup == "mcts_vs_raise":
    config.register_player(name="MCTS_Agent", algorithm=MCTSPokerPlayer())
    config.register_player(name="RaiseBot", algorithm=RaisedPlayer())
elif matchup == "mcts_vs_mcts":
    config.register_player(name="MCTS_Agent_500", algorithm=MCTSPokerPlayer(iterations=500))
    config.register_player(name="MCTS_Agent_100", algorithm=MCTSPokerPlayer(iterations=100))


# config.register_player(name="f1", algorithm=RandomPlayer())
# config.register_player(name="FT2", algorithm=RaisedPlayer())

game_result = start_poker(config, verbose=1)


print("\n=== Game Results ===")
for player in game_result['players']:
    print(f"{player['name']}: Final stack = {player['stack']}")




