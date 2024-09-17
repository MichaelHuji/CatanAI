# import gym
# from stable_baselines3 import PPO
import random

import gymnasium as gym
import numpy as np
import torch
import random
import torch
import torch.nn as nn
from gymnasium import Env

from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES, LandTile, BASE_MAP_TEMPLATE
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.models.player import Player
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, Action, ActionType
# from catanatron_gym.envs.catanatron_env import from_action_space
from catanatron_gym.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
)

from catanatron_gym.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
)

#   vvv   Michael's imports for reward function   vvv
from collections import defaultdict
from typing import Any, List, Tuple, Dict, Iterable

from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.board import Board
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    MONOPOLY,
    RESOURCES,
    YEAR_OF_PLENTY,
    SETTLEMENT,
    CITY,
    Action,
    ActionPrompt,
    ActionType,
)
from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    draw_from_listdeck,
    freqdeck_add,
    freqdeck_can_draw,
    freqdeck_contains,
    freqdeck_draw,
    freqdeck_from_listdeck,
    freqdeck_replenish,
    freqdeck_subtract,
    starting_devcard_bank,
    starting_resource_bank,
)
from catanatron.models.actions import (
    generate_playable_actions,
    road_building_possibilities,
    settlement_possibilities,
    city_possibilities
)
from catanatron.state import resource
from catanatron.state_functions import (
    get_player_buildings,
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    maintain_longest_road,
    play_dev_card,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_clean_turn,
    player_freqdeck_add,
    player_deck_draw,
    player_deck_random_draw,
    player_deck_replenish,
    player_freqdeck_subtract,
    player_deck_to_array,
    player_key,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)

from catanatron.models.player import Color, Player
from catanatron.models.enums import FastResource
#   ^^^   Michael's imports for reward function   ^^^

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges, Board
from catanatron.state_functions import player_key, player_has_rolled
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

#
# from MichaelFiles.myplayers import MyNNPlayer
# from MichaelFiles.myplayers import generate_x



from catanatron_gym.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)

from catanatron.game import Game
from catanatron.models.map import number_probability



# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Color
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.player import Player
from catanatron.game import Game
from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player


import math
import time
from collections import defaultdict

import numpy as np

from catanatron.game import Game
from catanatron.models.player import Player


def calculate_resource_production_for_number(board, number):
    """Computes resource payouts for given board and dice roll number.

    Args:
        board (Board): Board state
        resource_freqdeck (List[int]): Bank's resource freqdeck
        number (int): Sum of dice roll

    Returns:
        (dict, List[int]): 2-tuple.
            First element is color => freqdeck mapping. e.g. {Color.RED: [0,0,0,3,0]}.
            Second is an array of resources that couldn't be yieleded
            because they depleted.
    """

    intented_payout: Dict[Color, Dict[FastResource, int]] = defaultdict(
        lambda: defaultdict(int))
    for coordinate, tile in board.map.land_tiles.items():
        if tile.number != number:
            continue  # doesn't yield

        robber_penalty = 1
        if board.robber_coordinate == coordinate:
            robber_penalty = 0.9

        for node_id in tile.nodes.values():
            building = board.buildings.get(node_id, None)
            assert tile.resource is not None
            gain = 0
            if building is None:
                continue
            elif building[1] == SETTLEMENT:
                gain = 1
            elif building[1] == CITY:
                gain = 2
            intented_payout[building[0]][tile.resource] += (gain*robber_penalty)

    # build final data color => freqdeck structure
    payout = {}
    for player, player_payout in intented_payout.items():
        payout[player] = [0, 0, 0, 0, 0]

        for resource, count in player_payout.items():
            freqdeck_replenish(payout[player], count, resource)

    return payout

def calculate_resource_production_value_for_player(state, p0_color):
    # the probabilities for 6, 8 are not 5 because they have a higher chance to get blocked
    prob_dict = {2: 1, 3: 2, 4: 2.8, 5: 3.6, 6: 4.4, 7: 0, 8: 4.4, 9: 3.6, 10: 2.8, 11: 2, 12: 1}
    p0_total_payout = [0, 0, 0, 0, 0]
    for number, prob in prob_dict.items():
        payout = calculate_resource_production_for_number(state.board, number)
        # print(payout)
        if p0_color in payout.keys():
            p0_payout = payout[p0_color]
        else:
            p0_payout = [0, 0, 0, 0, 0]
        for i, amount in enumerate(p0_payout):
            if amount == 0:
                continue
            p0_total_payout[i] += (((amount+1)/2) * prob)  # 4/36 prob to get 1 is better than 2/36 to get 2
            p0_total_payout[i] += 0.5 # reward diversity in numbers

    production_value = 0
    for p_val in p0_total_payout:
        if p_val <= 0:
            production_value -= 3   # penalty for lack of resource diversity
        else:
            production_value += p_val
    return production_value





# from catanatron_experimental.machine_learning.players.playouts import run_playout
# from catanatron_experimental.machine_learning.players.tree_search_utils import (
#     execute_spectrum,
#     list_prunned_actions,
# )

from catanatron_experimental.cli.cli_players import register_player


# import random
#
# from catanatron.state_functions import (
#     get_longest_road_length,
#     get_played_dev_cards,
#     player_key,
#     player_num_dev_cards,
#     player_num_resource_cards,
# )
# from catanatron.models.player import Player, Color
# from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
# from catanatron_gym.envs import CatanatronEnv
# from catanatron_gym.features import (
#     build_production_features,
#     reachability_features,
#     resource_hand_features,
# )
#
# TRANSLATE_VARIETY = 4  # i.e. each new resource is like 4 production points
#
# DEFAULT_WEIGHTS = {
#     # Where to place. Note winning is best at all costs
#     "public_vps": 3e14,
#     "production": 1e8,
#     "enemy_production": -1e8,
#     "num_tiles": 1,
#     # Towards where to expand and when
#     "reachable_production_0": 0,
#     "reachable_production_1": 1e4,
#     "buildable_nodes": 1e3,
#     "longest_road": 10,
#     # Hand, when to hold and when to use.
#     "hand_synergy": 1e2,
#     "hand_resources": 1,
#     "discard_penalty": -5,
#     "hand_devs": 10,
#     "army_size": 10.1,
# }
#
# # Change these to play around with new values
# CONTENDER_WEIGHTS = {
#     "public_vps": 300000000000001.94,
#     "production": 100000002.04188395,
#     "enemy_production": -99999998.03389844,
#     "num_tiles": 2.91440418,
#     "reachable_production_0": 2.03820085,
#     "reachable_production_1": 10002.018773150001,
#     "buildable_nodes": 1001.86278466,
#     "longest_road": 12.127388499999999,
#     "hand_synergy": 102.40606877,
#     "hand_resources": 2.43644327,
#     "discard_penalty": -3.00141993,
#     "hand_devs": 10.721669799999999,
#     "army_size": 12.93844622,
# }
#
#
# def base_fn(params=DEFAULT_WEIGHTS):
#     def fn(game, p0_color):
#         production_features = build_production_features(True)
#         our_production_sample = production_features(game, p0_color)
#         enemy_production_sample = production_features(game, p0_color)
#         production = value_production(our_production_sample, "P0")
#         enemy_production = value_production(enemy_production_sample, "P1", False)
#
#         key = player_key(game.state, p0_color)
#         longest_road_length = get_longest_road_length(game.state, p0_color)
#
#         reachability_sample = reachability_features(game, p0_color, 2)
#         features = [f"P0_0_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
#         reachable_production_at_zero = sum([reachability_sample[f] for f in features])
#         features = [f"P0_1_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
#         reachable_production_at_one = sum([reachability_sample[f] for f in features])
#
#         hand_sample = resource_hand_features(game, p0_color)
#         features = [f"P0_{resource}_IN_HAND" for resource in RESOURCES]
#         distance_to_city = (
#             max(2 - hand_sample["P0_WHEAT_IN_HAND"], 0)
#             + max(3 - hand_sample["P0_ORE_IN_HAND"], 0)
#         ) / 5.0  # 0 means good. 1 means bad.
#         distance_to_settlement = (
#             max(1 - hand_sample["P0_WHEAT_IN_HAND"], 0)
#             + max(1 - hand_sample["P0_SHEEP_IN_HAND"], 0)
#             + max(1 - hand_sample["P0_BRICK_IN_HAND"], 0)
#             + max(1 - hand_sample["P0_WOOD_IN_HAND"], 0)
#         ) / 4.0  # 0 means good. 1 means bad.
#         hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2
#
#         num_in_hand = player_num_resource_cards(game.state, p0_color)
#         discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0
#
#         # blockability
#         buildings = game.state.buildings_by_color[p0_color]
#         owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
#         owned_tiles = set()
#         for n in owned_nodes:
#             owned_tiles.update(game.state.board.map.adjacent_tiles[n])
#         num_tiles = len(owned_tiles)
#
#         # TODO: Simplify to linear(?)
#         num_buildable_nodes = len(game.state.board.buildable_node_ids(p0_color))
#         longest_road_factor = (
#             params["longest_road"] if num_buildable_nodes == 0 else 0.1
#         )
#
#         return float(
#             game.state.player_state[f"{key}_VICTORY_POINTS"] * params["public_vps"]
#             + production * params["production"]
#             + enemy_production * params["enemy_production"]
#             + reachable_production_at_zero * params["reachable_production_0"]
#             + reachable_production_at_one * params["reachable_production_1"]
#             + hand_synergy * params["hand_synergy"]
#             + num_buildable_nodes * params["buildable_nodes"]
#             + num_tiles * params["num_tiles"]
#             + num_in_hand * params["hand_resources"]
#             + discard_penalty
#             + longest_road_length * longest_road_factor
#             + player_num_dev_cards(game.state, p0_color) * params["hand_devs"]
#             + get_played_dev_cards(game.state, p0_color, "KNIGHT") * params["army_size"]
#         )
#
#     return fn
#
#
# def value_production(sample, player_name="P0", include_variety=True):
#     proba_point = 2.778 / 100
#     features = [
#         f"EFFECTIVE_{player_name}_WHEAT_PRODUCTION",
#         f"EFFECTIVE_{player_name}_ORE_PRODUCTION",
#         f"EFFECTIVE_{player_name}_SHEEP_PRODUCTION",
#         f"EFFECTIVE_{player_name}_WOOD_PRODUCTION",
#         f"EFFECTIVE_{player_name}_BRICK_PRODUCTION",
#     ]
#     prod_sum = sum([sample[f] for f in features])
#     prod_variety = (
#         sum([sample[f] != 0 for f in features]) * TRANSLATE_VARIETY * proba_point
#     )
#     return prod_sum + (0 if not include_variety else prod_variety)
#
#
# def contender_fn(params):
#     return base_fn(params or CONTENDER_WEIGHTS)
#
#
# class ValueFunctionPlayer(Player):
#     """
#     Player that selects the move that maximizes a heuristic value function.
#
#     For now, the base value function only considers 1 enemy player.
#     """
#
#     def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):
#         super().__init__(color, is_bot)
#         self.value_fn_builder_name = (
#             "contender_fn" if value_fn_builder_name == "C" else "base_fn"
#         )
#         self.params = params
#         self.epsilon = epsilon
#
#     def decide(self, game, playable_actions):
#         if len(playable_actions) == 1:
#             return playable_actions[0]
#
#         # for exploration
#         if self.epsilon is not None and random.random() < self.epsilon:
#             return random.choice(playable_actions)
#
#         best_value = float("-inf")
#         best_action = None
#         for action in playable_actions:
#             game_copy = game.copy()
#             game_copy.execute(action)
#
#             value_fn = get_value_fn(self.value_fn_builder_name, self.params)
#             value = value_fn(game_copy, self.color)
#             if value > best_value:
#                 best_value = value
#                 best_action = action
#
#         return best_action
#
#     def __str__(self):
#         return super().__str__() + f"(value_fn={self.value_fn_builder_name})"
#
#
# def get_value_fn(name, params, value_function=None):
#     if value_function is not None:
#         return value_function
#     elif name == "base_fn":
#         return base_fn(DEFAULT_WEIGHTS)
#     elif name == "contender_fn":
#         return contender_fn(params)
#     else:
#         raise ValueError


# # Function to evaluate the trained model
# def evaluate_agent(env, model, episodes=100):
#     total_rewards = []
#     wins = 0
#     total_steps = []
#     victory_points = []
#     loss_vp = []
#     win_list = []
#     X_data = []
#     Y_data = []
#     total_non_random = 0
#     total_random = 0
#     for episode in range(episodes):
#         # obs = env.reset()
#         observation, info = env.reset()
#         done = False
#         total_reward = 0
#         steps = 0
#         vp = 0
#
#         win = 0
#         random_actions = 0
#         non_random_actions = 0
#
#         while not done:
#             # Get valid actions from the environment
#             valid_actions = env.get_valid_actions()
#
#             # play without exploration
#             action, _states = model.predict(observation, deterministic=True)
#
#             # if best action is not valid, try to get a different action from the model
#
#             for it in range(3):
#                 action, _states = model.predict(observation)
#                 if action in valid_actions:
#                     break
#
#             if action not in valid_actions:
#                 action, _states = model.predict(observation, deterministic=True)
#
#             if action not in valid_actions:
#                 for it in range(20):
#                     action, _states = model.predict(observation)
#                     if action in valid_actions:
#                         break
#
#             if action not in valid_actions:
#                 # If the action is invalid, select a random valid action
#                 # print("choosing random action")
#                 random_actions += 1
#                 action = np.random.choice(valid_actions)
#             else:
#                 non_random_actions += 1
#                 # print("action is not random")
#
#             # Take the action in the environment
#             observation, reward, terminated, truncated, info, X, Y = env.step(action)
#
#             X_data.append(X)
#             Y_data.append(Y)
#
#             done = terminated or truncated
#
#             # Accumulate rewards
#             total_reward += reward
#             steps += 1
#
#             # Check for win condition (depends on environment)
#             if 'WINNER' in info and info['WINNER']:
#                 wins += info['WINNER']
#                 win = 1
#
#             # Extract additional info such as victory points or game status
#             if 'VP' in info:
#                 vp = info['VP']
#                 # print(f"Victory Points: {victory_points}")
#
#         total_rewards.append(total_reward/steps)
#         total_steps.append(steps)
#         if vp < 10:
#             loss_vp.append(vp)
#         victory_points.append(vp)
#         win_list.append(win)
#         print(victory_points)
#         total_random +=random_actions
#         total_non_random += non_random_actions
#         print(f"random_actions = {random_actions}")
#         print(f"non random actions = {non_random_actions}")
#
#     # X_data = np.array(X_data)
#     # Y_data = np.array(Y_data)
#     np.savez(f"FvF_init_data_x_y_{episodes}games", X=np.array(X_data), Y=np.array(Y_data))
#
#     # Calculate metrics
#     avg_reward = np.mean(total_rewards)
#     win_rate = wins / episodes
#     avg_steps = np.mean(total_steps)
#
#     # print(f"Evaluation over {episodes} episodes:")
#     # print(f"Average Reward: {avg_reward}")
#     # print(total_rewards)
#     print(f"Win Rate: {win_rate * 100}%")
#     print(win_list)
#     # print(f"Average Game Length (in steps): {avg_steps}")
#     #
#     # print(f"Average victory points: {np.mean(victory_points)}")
#     # print(victory_points)
#     #
#     # print(f"Average loss victory points: {np.mean(loss_vp)}")
#     # print(loss_vp)
#     #
#     # print(f"total_random = {total_random}")
#     # print(f"total_non_random = {total_non_random}")
#     # print(f"total_random / total_non_random = {total_random / total_non_random}")

# Function to evaluate the trained model

def evaluate_agent(env: Env, model: Player, episodes=100):
    wins = 0
    X1_data = []
    Y_data = []
    X2_data = []
    X22_data = []
    x_turn_10_data = []
    x_turn_20_data = []
    x_turn_30_data = []

    print(model)
    print(env.enemies)

    for episode in range(episodes):

        observation, info = env.reset()
        done = False

        X1 = np.zeros(324)
        X2 = np.zeros(324)
        X22 = np.zeros(324)
        x_turn_10 = np.zeros(324)
        x_turn_20 = np.zeros(324)
        x_turn_30 = np.zeros(324)
        Y = 0
        while not done:

            action = model.decide(env.game, env.get_valid_actions())
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            # print("masked_ppo_eval")
            # print(observation)
            # print(reward)
            # print(terminated)
            # print(truncated)
            # print(info)
            X1 = info['x1']
            X2 = info['x2']
            X22 = info['x22']
            x_turn_10 = info['turn_10']
            x_turn_20 = info['turn_20']
            x_turn_30 = info['turn_30']
            Y = info['y']

        X1_data.append(X1)
        X2_data.append(X2)
        X22_data.append(X22)

        x_turn_10_data.append(x_turn_10)
        x_turn_20_data.append(x_turn_20)
        x_turn_30_data.append(x_turn_30)

        Y_data.append(Y)
        if len(Y_data)%10 == 0:
            print(len(Y_data))

    np.savez(f"game_simulation_data/ABvAB_1_{episodes}games_363features.npz", X=np.array(X1_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/ABvAB_2_{episodes}games_363features.npz", X=np.array(X2_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/ABvAB_22_{episodes}games_363features.npz", X=np.array(X22_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/ABvAB_turn_10_{episodes}games_363features.npz", X=np.array(x_turn_10_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/ABvAB_turn_20_{episodes}games_363features.npz", X=np.array(x_turn_20_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/ABvAB_turn_30_{episodes}games_363features.npz", X=np.array(x_turn_30_data), Y=np.array(Y_data))

    return np.array(X22_data), np.array(Y_data)

def mask_fn(env) -> np.ndarray:
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])






def calc_clean_prod(state, p_color):
    prob_dict = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    p0_total_payout = [0, 0, 0, 0, 0]
    # p0_nodes = [list(), list(), list(), list(), list()]
    for number, prob in prob_dict.items():
        payout = calculate_resource_production_for_number(state.board, number)
        if p_color in payout.keys():
            p0_payout = payout[p_color]
        else:
            p0_payout = [0, 0, 0, 0, 0]

        for i, amount in enumerate(p0_payout):
            # for j in range(int(amount)):
            #     p0_nodes[i].append(number)
            p0_total_payout[i] += (amount * prob)
    return p0_total_payout

def calc_dev_card_in_hand(state, p_key):
    dev_card_in_hand = 0
    dev_card_in_hand += state.player_state[f"{p_key}_KNIGHT_IN_HAND"]
    dev_card_in_hand += state.player_state[f"{p_key}_YEAR_OF_PLENTY_IN_HAND"]
    dev_card_in_hand += state.player_state[f"{p_key}_ROAD_BUILDING_IN_HAND"]
    dev_card_in_hand += state.player_state[f"{p_key}_MONOPOLY_IN_HAND"]
    dev_card_in_hand += state.player_state[f"{p_key}_VICTORY_POINT_IN_HAND"]

    return dev_card_in_hand

def generate_x(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    state = game.state
    board = state.board
    player_state = state.player_state
    # # we will later add port features for each node
    # X = np.zeros(648)
    # for i in range(54):
    #
    #     if i in get_player_buildings(state, p0_color, SETTLEMENT):
    #         X[12 * i] = 1
    #     elif i in get_player_buildings(state, p1_color, SETTLEMENT):
    #         X[12 * i] = -1
    #
    #     for j, resource in enumerate(RESOURCES):
    #         entry_index = 12*i + j + 1
    #         X[entry_index] = get_node_production(game.state.board.map, i, resource)
    #
    # return X


    # p0_buildings = get_player_buildings(state, p0_color, SETTLEMENT)
    # p1_buildings = get_player_buildings(state, p1_color, SETTLEMENT)
    #
    # X = np.zeros(324)
    # for i in range(54):
    #
    #     if i in p0_buildings:
    #         X[6 * i] = 1
    #     elif i in p1_buildings:
    #         X[6 * i] = -1
    #
    #     for j, resource in enumerate(RESOURCES):
    #         X[(6 * i) + (j + 1)] = get_node_production(game.state.board.map, i, resource)
    #
    # return X

    p0_settle = get_player_buildings(state, p0_color, SETTLEMENT)
    p1_settle = get_player_buildings(state, p1_color, SETTLEMENT)
    p0_city = get_player_buildings(state, p0_color, CITY)
    p1_city = get_player_buildings(state, p1_color, CITY)

    X = np.zeros(363)
    for i in range(54):

        if i in p0_settle:
            X[6 * i] = 1
        if i in p1_settle:
            X[6 * i] = -1
        if i in p0_city:
            X[6 * i] = 2
        if i in p1_city:
            X[6 * i] = -2

        for j, resource in enumerate(RESOURCES):
            X[(6 * i) + (j + 1)] = get_node_production(game.state.board.map, i, resource)

    # features for player 0 (BLUE / me)
    X[324] = player_state[f"{p_key}_VICTORY_POINTS"]
    X[325] = player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"]
    X[326] = player_state[f"{p_key}_CITIES_AVAILABLE"]
    X[327] = player_state[f"{p_key}_ROADS_AVAILABLE"] / 13
    X[328] = player_state[f"{p_key}_PLAYED_KNIGHT"]
    X[329] = player_state[f"{p_key}_HAS_ARMY"]
    X[330] = player_state[f"{p_key}_HAS_ROAD"]
    X[331] = player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    X[332] = calc_dev_card_in_hand(state, p_key)
    for j, r in enumerate(RESOURCES):
        X[333 + j] = player_state[f"{p_key}_{r}_IN_HAND"]
    p0_prod = calc_clean_prod(game.state, p0_color)
    for j, p in enumerate(p0_prod):
        X[338+j] = p

    # features for player 1 (RED / enemy)
    X[343] = player_state[f"{p1_key}_VICTORY_POINTS"]
    X[344] = player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"]
    X[345] = player_state[f"{p1_key}_CITIES_AVAILABLE"]
    X[346] = player_state[f"{p1_key}_ROADS_AVAILABLE"] / 13
    X[347] = player_state[f"{p1_key}_PLAYED_KNIGHT"]
    X[348] = player_state[f"{p1_key}_HAS_ARMY"]
    X[349] = player_state[f"{p1_key}_HAS_ROAD"]
    X[350] = player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
    X[351] = calc_dev_card_in_hand(state, p1_key)
    for j, r in enumerate(RESOURCES):
        X[352 + j] = player_state[f"{p1_key}_{r}_IN_HAND"]
    p1_prod = calc_clean_prod(game.state, p1_color)
    for j, p in enumerate(p1_prod):
        X[357 + j] = p

    X[362] = state.num_turns

    return X

def get_node_production(catan_map, node_id, resource):

    prob_dict = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    production = 0
    for tile in catan_map.adjacent_tiles[node_id]:
        if tile.resource == resource:
            production += prob_dict[tile.number]

    return production

    # tiles = catan_map.adjacent_tiles[node_id]
    # return sum([number_probability(t.number) for t in tiles if t.resource == resource])


# from catanatron_gym.envs.catanatron_env import generate_x
# from LearnModel2 import Net


class Net(nn.Module):
    def __init__(self, input_size=363):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.output = nn.Linear(32, 1)

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)  # Optional dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.output(x))
        return x


BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology

TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]

ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    # TODO: One for each tile (and abuse 1v1 setting).
    *[(ActionType.MOVE_ROBBER, tile) for tile in TILE_COORDINATES],
    (ActionType.DISCARD, None),
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
        for i, first_card in enumerate(RESOURCES)
        for j in range(i, len(RESOURCES))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCES],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
    # 4:1 with bank
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 3:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 2:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    (ActionType.END_TURN, None),
]

def normalize_action(action):
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    return normalized

def to_action_space(action):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))

def from_action_space(action_int, playable_actions):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches ACTIONS_ARRAY blueprint

    # action_int = to_action_space(action_int)
    (action_type, value) = ACTIONS_ARRAY[action_int]
    catan_action = None
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            catan_action = action
            break  # return the first one
    assert catan_action is not None
    return catan_action

#
# @register_player("NN")
class MymaskNNPlayer(Player):
    """
    Player that chooses actions by maximizing Victory Points greedily.
    If multiple actions lead to the same max-points-achievable
    in this turn, selects from them at random.
    """

    def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):
        super().__init__(color, is_bot)
        # self.model = torch.load('FvF_66K_363feat_model.pth')
        #
        # # Make sure to call model.eval() if you're in inference mode
        # self.model.eval()
        self.model = Net()

        # Load the state_dict

        # torch.save(model.state_dict(), )
        # self.model.load_state_dict(torch.load('FvF_22VP_114K_363feat_model_weights.pth'))
        # self.model.load_state_dict(torch.load('FvF_turn_20_114K_363feat_model_weights.pth'))


        # self.model.load_state_dict(torch.load(f'FvF_all_129K_363feat_model_weights_epoch39.pth'))

        # self.model.load_state_dict(torch.load(f'363_64_32_16_1_FvF_all_129K_363feat_model_weights_epoch29.pth'))

        self.model.load_state_dict(torch.load(f'363_64_32_16_1_FvF_all_129K_363feat_model_weights_epoch29.pth'))
        # self.model.load_state_dict(torch.load(f'FvF_all_129K_363feat_model_weights_epoch47.pth'))

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_actions = []
        for action in playable_actions:
            game_copy = game.copy()
            if isinstance(action, int):
                catan_action = from_action_space(action, game.state.playable_actions)
                game_copy.execute(catan_action)

            else:
                game_copy.execute(action)
            # game_copy.execute(action)

            action_vector = generate_x(game_copy, self.color)

            # key = player_key(game_copy.state, self.color)
            action_value = self.model(torch.tensor(action_vector, dtype=torch.float32))
            if action_value == best_value:
                best_actions.append(action)
            if action_value > best_value:
                best_value = action_value
                best_actions = [action]

        return random.choice(best_actions)



# Init Environment and Model
env = gym.make("catanatron_gym:catanatron-v1")
# env = ActionMasker(env, mask_fn)  # Wrap to enable masking
# model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Create the environment
# env = gym.make('Catanatron-v0')
# env = gym.make("catanatron_gym:catanatron-v1")
# timesteps = 2
# timesteps = 1000000
# model = MaskablePPO.load(f"masked_ppo_{timesteps}", env=env)
# model = MaskablePPO.load(f"masked_ppo_{timesteps}_r", env=env)


# model = ValueFunctionPlayer(Color.BLUE)
# model = MymaskNNPlayer(Color.BLUE)
model = AlphaBetaPlayer(Color.BLUE)
# model = RandomPlayer(Color.BLUE)
# model = WeightedRandomPlayer(Color.BLUE)


X_data, Y_data = evaluate_agent(env, model, episodes=700)  # generate data for {episodes} games

Y_data[Y_data == -1] = 0

print(np.count_nonzero(Y_data))

