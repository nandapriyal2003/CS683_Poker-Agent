import sys
sys.path.insert(0, './pypokerengine/api/')
import game
setup_config = game.setup_config
start_poker = game.start_poker
import time
from argparse import ArgumentParser
import importlib


""" =========== *Remember to import your agent!!! =========== """
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from mcts_player import MCTSPokerPlayer
""" ========================================================= """

def get_agent_class(agent_name):
    """Dynamically import and return the agent class based on name"""
    if agent_name == "RandomPlayer":
        return RandomPlayer
    elif agent_name == "RaisedPlayer":
        return RaisedPlayer
    elif agent_name == "MCTSPokerPlayer":
        return MCTSPokerPlayer
    else:
        # Dynamically import
        try:
            module_name, class_name = agent_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except:
            return RandomPlayer  # Default fallback


def testperf(agent_name1, agent1_name, agent_name2, agent2_name, agent1_params={}, agent2_params={}):		
    # Init to play games
    num_game = 10  # 500
    max_round = 100  # 1000
    initial_stack = 10000
    smallblind_amount = 20

    # Get agent classes
    agent1_class = get_agent_class(agent1_name)
    agent2_class = get_agent_class(agent2_name)
    
    # Initialize agents with parameters
    if agent1_class == MCTSPokerPlayer:
        agent1 = agent1_class(**agent1_params)
    else:
        agent1 = agent1_class()
        
    if agent2_class == MCTSPokerPlayer:
        agent2 = agent2_class(**agent2_params)
    else:
        agent2 = agent2_class()
    
    # Init pot of players
    agent1_pot = 0
    agent2_pot = 0

    # Setting configuration
    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=smallblind_amount)
    
    # Register players
    config.register_player(name=agent_name1, algorithm=agent1)
    config.register_player(name=agent_name2, algorithm=agent2)
    
    print(f"\n=== Starting tournament: {agent_name1} vs {agent_name2} ===")
    print(f"Playing {num_game} games of {max_round} rounds each\n")
    
    # Start playing num_game games
    for game_num in range(1, num_game+1):
        print(f"Game number: {game_num}/{num_game}")
        try:
            game_result = start_poker(config, verbose=0)
            agent1_pot += game_result['players'][0]['stack']
            agent2_pot += game_result['players'][1]['stack']
        except Exception as e:
            print(f"Error in game {game_num}: {e}")
            continue

    print("\n" + "="*50)
    print(f"FINAL RESULTS after {num_game} games:")
    print("="*50)
    print(f"\n{agent_name1}'s final pot: {agent1_pot}")
    print(f"{agent_name2}'s final pot: {agent2_pot}")
    print(f"Difference: {agent1_pot - agent2_pot} chips")

    if agent1_pot < agent2_pot:
        print(f"\nCongratulations! {agent_name2} has won!")
    elif agent1_pot > agent2_pot:
        print(f"\nCongratulations! {agent_name1} has won!")
    else:
        print("\nIt's a draw!") 


def parse_arguments():
    parser = ArgumentParser(description="Test poker agents against each other")
    parser.add_argument('-n1', '--agent_name1', help="Name of agent 1", default="MCTS_Agent", type=str)
    parser.add_argument('-a1', '--agent1', help="Agent 1 class name", default="MCTSPokerPlayer", type=str)
    parser.add_argument('-n2', '--agent_name2', help="Name of agent 2", default="Random_Player", type=str)
    parser.add_argument('-a2', '--agent2', help="Agent 2 class name", default="RandomPlayer", type=str)
    parser.add_argument('-i1', '--iterations1', default=100, type=int, help="MCTS iterations for agent1")
    parser.add_argument('-i2', '--iterations2', default=50, type=int, help="MCTS iterations for agent2")
    args = parser.parse_args()
    return args.agent_name1, args.agent1, args.agent_name2, args.agent2, args.iterations1, args.iterations2

if __name__ == '__main__':
    name1, agent1_name, name2, agent2_name, it1, it2 = parse_arguments()
    
    # Prepare parameters for MCTS agents
    params1 = {'iterations': it1} if agent1_name == "MCTSPokerPlayer" else {}
    params2 = {'iterations': it2} if agent2_name == "MCTSPokerPlayer" else {}
    
    start = time.time()
    testperf(name1, agent1_name, name2, agent2_name, params1, params2)
    end = time.time()

    print(f"\nTime taken to play: {(end-start):.2f} seconds")
    print(f"Average time per game: {(end-start)/10:.2f} seconds")


