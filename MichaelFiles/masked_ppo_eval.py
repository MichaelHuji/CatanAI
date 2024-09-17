# # import gym
# # from stable_baselines3 import PPO
#
# import gymnasium as gym
# import numpy as np
# from gymnasium import Env
# # from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# # from sb3_contrib.common.wrappers import ActionMasker
# # from sb3_contrib.ppo_mask import MaskablePPO
#
# from catanatron import Color
# from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
# from catanatron.players.weighted_random import WeightedRandomPlayer
# from catanatron.models.player import Color, Player, RandomPlayer
# # from myplayers import MyNNPlayer
# # from catanatron.players.myplayers import MyNNPlayer
# from MichaelFiles.myplayers import MyNNPlayer
#
#
# # import random
# #
# # from catanatron.state_functions import (
# #     get_longest_road_length,
# #     get_played_dev_cards,
# #     player_key,
# #     player_num_dev_cards,
# #     player_num_resource_cards,
# # )
# # from catanatron.models.player import Player, Color
# # from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
# # from catanatron_gym.envs import CatanatronEnv
# # from catanatron_gym.features import (
# #     build_production_features,
# #     reachability_features,
# #     resource_hand_features,
# # )
# #
# # TRANSLATE_VARIETY = 4  # i.e. each new resource is like 4 production points
# #
# # DEFAULT_WEIGHTS = {
# #     # Where to place. Note winning is best at all costs
# #     "public_vps": 3e14,
# #     "production": 1e8,
# #     "enemy_production": -1e8,
# #     "num_tiles": 1,
# #     # Towards where to expand and when
# #     "reachable_production_0": 0,
# #     "reachable_production_1": 1e4,
# #     "buildable_nodes": 1e3,
# #     "longest_road": 10,
# #     # Hand, when to hold and when to use.
# #     "hand_synergy": 1e2,
# #     "hand_resources": 1,
# #     "discard_penalty": -5,
# #     "hand_devs": 10,
# #     "army_size": 10.1,
# # }
# #
# # # Change these to play around with new values
# # CONTENDER_WEIGHTS = {
# #     "public_vps": 300000000000001.94,
# #     "production": 100000002.04188395,
# #     "enemy_production": -99999998.03389844,
# #     "num_tiles": 2.91440418,
# #     "reachable_production_0": 2.03820085,
# #     "reachable_production_1": 10002.018773150001,
# #     "buildable_nodes": 1001.86278466,
# #     "longest_road": 12.127388499999999,
# #     "hand_synergy": 102.40606877,
# #     "hand_resources": 2.43644327,
# #     "discard_penalty": -3.00141993,
# #     "hand_devs": 10.721669799999999,
# #     "army_size": 12.93844622,
# # }
# #
# #
# # def base_fn(params=DEFAULT_WEIGHTS):
# #     def fn(game, p0_color):
# #         production_features = build_production_features(True)
# #         our_production_sample = production_features(game, p0_color)
# #         enemy_production_sample = production_features(game, p0_color)
# #         production = value_production(our_production_sample, "P0")
# #         enemy_production = value_production(enemy_production_sample, "P1", False)
# #
# #         key = player_key(game.state, p0_color)
# #         longest_road_length = get_longest_road_length(game.state, p0_color)
# #
# #         reachability_sample = reachability_features(game, p0_color, 2)
# #         features = [f"P0_0_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
# #         reachable_production_at_zero = sum([reachability_sample[f] for f in features])
# #         features = [f"P0_1_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
# #         reachable_production_at_one = sum([reachability_sample[f] for f in features])
# #
# #         hand_sample = resource_hand_features(game, p0_color)
# #         features = [f"P0_{resource}_IN_HAND" for resource in RESOURCES]
# #         distance_to_city = (
# #             max(2 - hand_sample["P0_WHEAT_IN_HAND"], 0)
# #             + max(3 - hand_sample["P0_ORE_IN_HAND"], 0)
# #         ) / 5.0  # 0 means good. 1 means bad.
# #         distance_to_settlement = (
# #             max(1 - hand_sample["P0_WHEAT_IN_HAND"], 0)
# #             + max(1 - hand_sample["P0_SHEEP_IN_HAND"], 0)
# #             + max(1 - hand_sample["P0_BRICK_IN_HAND"], 0)
# #             + max(1 - hand_sample["P0_WOOD_IN_HAND"], 0)
# #         ) / 4.0  # 0 means good. 1 means bad.
# #         hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2
# #
# #         num_in_hand = player_num_resource_cards(game.state, p0_color)
# #         discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0
# #
# #         # blockability
# #         buildings = game.state.buildings_by_color[p0_color]
# #         owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
# #         owned_tiles = set()
# #         for n in owned_nodes:
# #             owned_tiles.update(game.state.board.map.adjacent_tiles[n])
# #         num_tiles = len(owned_tiles)
# #
# #         # TODO: Simplify to linear(?)
# #         num_buildable_nodes = len(game.state.board.buildable_node_ids(p0_color))
# #         longest_road_factor = (
# #             params["longest_road"] if num_buildable_nodes == 0 else 0.1
# #         )
# #
# #         return float(
# #             game.state.player_state[f"{key}_VICTORY_POINTS"] * params["public_vps"]
# #             + production * params["production"]
# #             + enemy_production * params["enemy_production"]
# #             + reachable_production_at_zero * params["reachable_production_0"]
# #             + reachable_production_at_one * params["reachable_production_1"]
# #             + hand_synergy * params["hand_synergy"]
# #             + num_buildable_nodes * params["buildable_nodes"]
# #             + num_tiles * params["num_tiles"]
# #             + num_in_hand * params["hand_resources"]
# #             + discard_penalty
# #             + longest_road_length * longest_road_factor
# #             + player_num_dev_cards(game.state, p0_color) * params["hand_devs"]
# #             + get_played_dev_cards(game.state, p0_color, "KNIGHT") * params["army_size"]
# #         )
# #
# #     return fn
# #
# #
# # def value_production(sample, player_name="P0", include_variety=True):
# #     proba_point = 2.778 / 100
# #     features = [
# #         f"EFFECTIVE_{player_name}_WHEAT_PRODUCTION",
# #         f"EFFECTIVE_{player_name}_ORE_PRODUCTION",
# #         f"EFFECTIVE_{player_name}_SHEEP_PRODUCTION",
# #         f"EFFECTIVE_{player_name}_WOOD_PRODUCTION",
# #         f"EFFECTIVE_{player_name}_BRICK_PRODUCTION",
# #     ]
# #     prod_sum = sum([sample[f] for f in features])
# #     prod_variety = (
# #         sum([sample[f] != 0 for f in features]) * TRANSLATE_VARIETY * proba_point
# #     )
# #     return prod_sum + (0 if not include_variety else prod_variety)
# #
# #
# # def contender_fn(params):
# #     return base_fn(params or CONTENDER_WEIGHTS)
# #
# #
# # class ValueFunctionPlayer(Player):
# #     """
# #     Player that selects the move that maximizes a heuristic value function.
# #
# #     For now, the base value function only considers 1 enemy player.
# #     """
# #
# #     def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):
# #         super().__init__(color, is_bot)
# #         self.value_fn_builder_name = (
# #             "contender_fn" if value_fn_builder_name == "C" else "base_fn"
# #         )
# #         self.params = params
# #         self.epsilon = epsilon
# #
# #     def decide(self, game, playable_actions):
# #         if len(playable_actions) == 1:
# #             return playable_actions[0]
# #
# #         # for exploration
# #         if self.epsilon is not None and random.random() < self.epsilon:
# #             return random.choice(playable_actions)
# #
# #         best_value = float("-inf")
# #         best_action = None
# #         for action in playable_actions:
# #             game_copy = game.copy()
# #             game_copy.execute(action)
# #
# #             value_fn = get_value_fn(self.value_fn_builder_name, self.params)
# #             value = value_fn(game_copy, self.color)
# #             if value > best_value:
# #                 best_value = value
# #                 best_action = action
# #
# #         return best_action
# #
# #     def __str__(self):
# #         return super().__str__() + f"(value_fn={self.value_fn_builder_name})"
# #
# #
# # def get_value_fn(name, params, value_function=None):
# #     if value_function is not None:
# #         return value_function
# #     elif name == "base_fn":
# #         return base_fn(DEFAULT_WEIGHTS)
# #     elif name == "contender_fn":
# #         return contender_fn(params)
# #     else:
# #         raise ValueError
#
#
# # # Function to evaluate the trained model
# # def evaluate_agent(env, model, episodes=100):
# #     total_rewards = []
# #     wins = 0
# #     total_steps = []
# #     victory_points = []
# #     loss_vp = []
# #     win_list = []
# #     X_data = []
# #     Y_data = []
# #     total_non_random = 0
# #     total_random = 0
# #     for episode in range(episodes):
# #         # obs = env.reset()
# #         observation, info = env.reset()
# #         done = False
# #         total_reward = 0
# #         steps = 0
# #         vp = 0
# #
# #         win = 0
# #         random_actions = 0
# #         non_random_actions = 0
# #
# #         while not done:
# #             # Get valid actions from the environment
# #             valid_actions = env.get_valid_actions()
# #
# #             # play without exploration
# #             action, _states = model.predict(observation, deterministic=True)
# #
# #             # if best action is not valid, try to get a different action from the model
# #
# #             for it in range(3):
# #                 action, _states = model.predict(observation)
# #                 if action in valid_actions:
# #                     break
# #
# #             if action not in valid_actions:
# #                 action, _states = model.predict(observation, deterministic=True)
# #
# #             if action not in valid_actions:
# #                 for it in range(20):
# #                     action, _states = model.predict(observation)
# #                     if action in valid_actions:
# #                         break
# #
# #             if action not in valid_actions:
# #                 # If the action is invalid, select a random valid action
# #                 # print("choosing random action")
# #                 random_actions += 1
# #                 action = np.random.choice(valid_actions)
# #             else:
# #                 non_random_actions += 1
# #                 # print("action is not random")
# #
# #             # Take the action in the environment
# #             observation, reward, terminated, truncated, info, X, Y = env.step(action)
# #
# #             X_data.append(X)
# #             Y_data.append(Y)
# #
# #             done = terminated or truncated
# #
# #             # Accumulate rewards
# #             total_reward += reward
# #             steps += 1
# #
# #             # Check for win condition (depends on environment)
# #             if 'WINNER' in info and info['WINNER']:
# #                 wins += info['WINNER']
# #                 win = 1
# #
# #             # Extract additional info such as victory points or game status
# #             if 'VP' in info:
# #                 vp = info['VP']
# #                 # print(f"Victory Points: {victory_points}")
# #
# #         total_rewards.append(total_reward/steps)
# #         total_steps.append(steps)
# #         if vp < 10:
# #             loss_vp.append(vp)
# #         victory_points.append(vp)
# #         win_list.append(win)
# #         print(victory_points)
# #         total_random +=random_actions
# #         total_non_random += non_random_actions
# #         print(f"random_actions = {random_actions}")
# #         print(f"non random actions = {non_random_actions}")
# #
# #     # X_data = np.array(X_data)
# #     # Y_data = np.array(Y_data)
# #     np.savez(f"FvF_init_data_x_y_{episodes}games", X=np.array(X_data), Y=np.array(Y_data))
# #
# #     # Calculate metrics
# #     avg_reward = np.mean(total_rewards)
# #     win_rate = wins / episodes
# #     avg_steps = np.mean(total_steps)
# #
# #     # print(f"Evaluation over {episodes} episodes:")
# #     # print(f"Average Reward: {avg_reward}")
# #     # print(total_rewards)
# #     print(f"Win Rate: {win_rate * 100}%")
# #     print(win_list)
# #     # print(f"Average Game Length (in steps): {avg_steps}")
# #     #
# #     # print(f"Average victory points: {np.mean(victory_points)}")
# #     # print(victory_points)
# #     #
# #     # print(f"Average loss victory points: {np.mean(loss_vp)}")
# #     # print(loss_vp)
# #     #
# #     # print(f"total_random = {total_random}")
# #     # print(f"total_non_random = {total_non_random}")
# #     # print(f"total_random / total_non_random = {total_random / total_non_random}")
#
# # Function to evaluate the trained model
#
# def evaluate_agent(env: Env, model: Player, episodes=100):
#     wins = 0
#     X1_data = []
#     Y_data = []
#     X2_data = []
#     X22_data = []
#     x_turn_10_data = []
#     x_turn_20_data = []
#     x_turn_30_data = []
#
#     for episode in range(episodes):
#
#         observation, info = env.reset()
#         done = False
#
#         X1 = np.zeros(324)
#         X2 = np.zeros(324)
#         X22 = np.zeros(324)
#         x_turn_10 = np.zeros(324)
#         x_turn_20 = np.zeros(324)
#         x_turn_30 = np.zeros(324)
#         Y = 0
#         while not done:
#
#             action = model.decide(env.game, env.get_valid_actions())
#             observation, reward, terminated, truncated, info = env.step(action)
#
#             done = terminated or truncated
#             # print("masked_ppo_eval")
#             # print(observation)
#             # print(reward)
#             # print(terminated)
#             # print(truncated)
#             # print(info)
#             X1 = info['x1']
#             X2 = info['x2']
#             X22 = info['x22']
#             x_turn_10 = info['turn_10']
#             x_turn_20 = info['turn_20']
#             x_turn_30 = info['turn_30']
#             Y = info['y']
#
#         X1_data.append(X1)
#         X2_data.append(X2)
#         X22_data.append(X22)
#
#         x_turn_10_data.append(x_turn_10)
#         x_turn_20_data.append(x_turn_20)
#         x_turn_30_data.append(x_turn_30)
#
#         Y_data.append(Y)
#         if len(Y_data)%100 == 0:
#             print(len(Y_data))
#
#     np.savez(f"game_simulation_data/NN1vNN1_1_{episodes}games_363features.npz", X=np.array(X1_data), Y=np.array(Y_data))
#
#     np.savez(f"game_simulation_data/NN1vNN1_2_{episodes}games_363features.npz", X=np.array(X2_data), Y=np.array(Y_data))
#
#     np.savez(f"game_simulation_data/NN1vNN1_22_{episodes}games_363features.npz", X=np.array(X22_data), Y=np.array(Y_data))
#
#     np.savez(f"game_simulation_data/NN1vNN1_turn_10_{episodes}games_363features.npz", X=np.array(x_turn_10_data), Y=np.array(Y_data))
#
#     np.savez(f"game_simulation_data/NN1vNN1_turn_20_{episodes}games_363features.npz", X=np.array(x_turn_20_data), Y=np.array(Y_data))
#
#     np.savez(f"game_simulation_data/NN1vNN1_turn_30_{episodes}games_363features.npz", X=np.array(x_turn_30_data), Y=np.array(Y_data))
#
#     return np.array(X22_data), np.array(Y_data)
#
# def mask_fn(env) -> np.ndarray:
#     valid_actions = env.get_valid_actions()
#     mask = np.zeros(env.action_space.n, dtype=np.float32)
#     mask[valid_actions] = 1
#
#     return np.array([bool(i) for i in mask])
#
#
# # Init Environment and Model
# env = gym.make("catanatron_gym:catanatron-v1")
# # env = ActionMasker(env, mask_fn)  # Wrap to enable masking
# # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
#
# # Create the environment
# # env = gym.make('Catanatron-v0')
# # env = gym.make("catanatron_gym:catanatron-v1")
# # timesteps = 2
# # timesteps = 1000000
# # model = MaskablePPO.load(f"masked_ppo_{timesteps}", env=env)
# # model = MaskablePPO.load(f"masked_ppo_{timesteps}_r", env=env)
#
#
# # model = ValueFunctionPlayer(Color.BLUE)
# model = MyNNPlayer(Color.BLUE)
# # model = RandomPlayer(Color.BLUE)
# # model = WeightedRandomPlayer(Color.BLUE)
#
#
# X_data, Y_data = evaluate_agent(env, model, episodes=5000)  # generate data for {episodes} games
#
# Y_data[Y_data == -1] = 0
#
# print(np.count_nonzero(Y_data))
#
