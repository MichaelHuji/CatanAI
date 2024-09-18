import gymnasium as gym
from gymnasium import spaces
import numpy as np

from collections import defaultdict
from typing import List, Dict

from catanatron.models.enums import SETTLEMENT,CITY

from catanatron.models.decks import (freqdeck_replenish,
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


from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges, Board
from catanatron.state_functions import player_key, player_has_rolled
from catanatron.game import Game, TURNS_LIMIT


from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
# from catanatron.catanatron_experimental.catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

from catanatron_experimental.MichaelFiles.MyNNPlayer import MyNNPlayer
from catanatron_experimental.MichaelFiles.Features import generate_x
from catanatron_experimental.MichaelFiles.ActionsSpace import (to_action_space, from_action_space,
                                                               ACTION_SPACE_SIZE, ACTIONS_ARRAY, ACTION_TYPES)


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

FEATURES = get_feature_ordering(num_players=2)
NUM_FEATURES = len(FEATURES)


def to_action_type_space(action):
    return ACTION_TYPES.index(action.action_type)


# Highest features is NUM_RESOURCES_IN_HAND which in theory is all resource cards
HIGH = 19 * 5


def simple_reward(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return 0
    else:
        return -1



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


class CatanatronEnv(gym.Env):
    metadata = {"render_modes": []}

    action_space = spaces.Discrete(ACTION_SPACE_SIZE)
    # TODO: This could be smaller (there are many binary features). float b.c. TILE0_PROBA
    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    reward_range = (-1, 1)

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)

        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])
        # self.enemies = self.config.get("enemies", [WeightedRandomPlayer(Color.RED)])
        # self.enemies = self.config.get("enemies", [ValueFunctionPlayer(Color.RED)])
        # self.enemies = self.config.get("enemies", [AlphaBetaPlayer(Color.RED)])
        # self.enemies = self.config.get("enemies", [MyNNPlayer(Color.RED)])

        self.X_1VP = 0
        self.X_2VP = 0
        self.X_11VP = 0
        self.X_22VP = 0
        self.got_X_1VP = False
        self.got_X_2VP = False
        self.got_X_11VP = False
        self.got_X_22VP = False
        self.x_turn = {10: 0 , 20: 0 , 30: 0}
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 10

        # TODO: Make self.action_space smaller if possible (per map_type)
        # self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        try:
            if isinstance(action, int):
                catan_action = from_action_space(action, self.game.state.playable_actions)
            else:
                catan_action = action
            # catan_action = from_action_space(action, self.game.state.playable_actions)
        except Exception as e:
            self.invalid_actions_count += 1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )

            y = 0
            if winning_color == Color.BLUE:
                y = 1
            elif winning_color == Color.RED:
                y = -1

            info = dict(valid_actions=self.get_valid_actions())
            info['x1'] = self.X_1VP
            info['x2'] = self.X_2VP
            info['x22'] = self.X_22VP
            info['turn_10'] = self.x_turn[10]
            info['turn_20'] = self.x_turn[20]
            info['turn_30'] = self.x_turn[30]
            info['y'] = y
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        p1_color = Color.RED
        if self.p0.color == p1_color:
            p1_color = Color.BLUE
        p_key = player_key(self.game.state, self.p0.color)
        p1_key = player_key(self.game.state, p1_color)

        if (not self.got_X_1VP) and (self.game.state.player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"] == 1):
            self.X_1VP = generate_x(self.game, self.p0.color)
            self.got_X_1VP = True

        if ((not self.got_X_2VP) and (self.game.state.player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"] == 2)):
            self.X_2VP = generate_x(self.game, self.p0.color)
            self.got_X_2VP = True

        if ((not self.got_X_22VP) and (self.game.state.player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"] == 2) and
                (self.game.state.player_state[f"{p1_key}_ACTUAL_VICTORY_POINTS"] == 2)):
            self.X_22VP = generate_x(self.game, self.p0.color)
            self.got_X_22VP = True

        if (isinstance(self.x_turn[10], int)) and (self.game.state.num_turns >= 5 and self.game.state.num_turns < 14):
            self.x_turn[10] = generate_x(self.game, self.p0.color)

        if (isinstance(self.x_turn[20], int)) and (self.game.state.num_turns >= 14 and self.game.state.num_turns < 21):
            self.x_turn[20] = generate_x(self.game, self.p0.color)

        if (isinstance(self.x_turn[30], int)) and (self.game.state.num_turns >= 22):
            self.x_turn[30] = generate_x(self.game, self.p0.color)

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())
        winning_color = self.game.winning_color()

        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)

        y = 0
        if winning_color == Color.BLUE:
            y = 1
        elif winning_color == Color.RED:
            y = -1

        info['x1'] = self.X_1VP
        info['x2'] = self.X_2VP
        info['x22'] = self.X_22VP
        info['turn_10'] = self.x_turn[10]
        info['turn_20'] = self.x_turn[20]
        info['turn_30'] = self.x_turn[30]
        info['y'] = y
        return observation, reward, terminated, truncated, info

    def reset(self,seed=None,options=None,):
        super().reset(seed=seed)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        self.game = Game(players=self.players,seed=seed,catan_map=catan_map,vps_to_win=self.vps_to_win,)
        self.invalid_actions_count = 0

        self.X_1VP = np.zeros(363)
        self.got_X_1VP = False
        self.X_2VP = np.zeros(363)
        self.got_X_2VP = False
        self.X_22VP = np.zeros(363)
        self.got_X_22VP = False
        self.x_turn = {10: 0 , 20: 0 , 30: 0}

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot


CatanatronEnv.__doc__ = f"""
1v1 environment against a random player

Attributes:
    reward_range: -1 if player lost, 1 if player won, 0 otherwise.
    action_space: Integers from the [0, 289] interval. 
        See Action Space table below.
    observation_space: Numeric Feature Vector. See Observation Space table 
        below for quantities. They appear in vector in alphabetical order,
        from the perspective of "current" player (hiding/showing information
        accordingly). P0 is "current" player. P1 is next in line.
        
        We use the following nomenclature for Tile ids and Node ids.
        Edge ids are self-describing (node-id, node-id) tuples. We also
        use Cube coordinates for tiles (see 
        https://www.redblobgames.com/grids/hexagons/#coordinates)

.. image:: _static/tile-ids.png
  :width: 300
  :alt: Tile Ids
.. image:: _static/node-ids.png
  :width: 300
  :alt: Node Ids

.. list-table:: Action Space
   :widths: 10 100
   :header-rows: 1

   * - Integer
     - Catanatron Action
"""
for i, v in enumerate(ACTIONS_ARRAY):
    CatanatronEnv.__doc__ += f"   * - {i}\n     - {v}\n"

CatanatronEnv.__doc__ += """

.. list-table:: Observation Space (Raw)
   :widths: 10 50 10 10
   :header-rows: 1

   * - Feature Name
     - Description
     - Number of Features (N=number of players)
     - Type

   * - BANK_<resource>
     - Number of cards of that `resource` in bank
     - 5
     - Integer
   * - BANK_DEV_CARDS
     - Number of development cards in bank
     - 1
     - Integer
    
   * - EDGE<i>_P<j>_ROAD
     - Whether edge `i` is owned by player `j`
     - 72 * N
     - Boolean
   * - NODE<i>_P<j>_SETTLEMENT
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - NODE<i>_P<j>_CITY
     - Whether player `j` has a city in node `i`
     - 54 * N
     - Boolean
   * - PORT<i>_IS_<resource>
     - Whether node `i` is port of `resource` (or THREE_TO_ONE).
     - 9 * 6
     - Boolean
   * - TILE<i>_HAS_ROBBER
     - Whether robber is on tile `i`.
     - 19
     - Boolean
   * - TILE<i>_IS_<resource>
     - Whether tile `i` yields `resource` (or DESERT).
     - 19 * 6
     - Boolean
   * - TILE<i>_PROBA
     - Tile `i`'s probability of being rolled.
     - 19
     - Float

   * - IS_DISCARDING
     - Whether current player must discard. For now, there is only 1 
       discarding action (at random), since otherwise action space
       would explode in size.
     - 1
     - Boolean
   * - IS_MOVING_ROBBER
     - Whether current player must move robber (because played knight
       or because rolled a 7).
     - 1
     - Boolean
   * - P<i>_HAS_ROLLED
     - Whether player `i` already rolled dice.
     - N
     - Boolean
   * - P0_HAS_PLAYED _DEVELOPMENT_CARD _IN_TURN
     - Whether current player already played a development card
     - 1
     - Boolean

   * - P0_ACTUAL_VPS
     - Total Victory Points (including Victory Point Development Cards)
     - 1
     - Integer
   * - P0_<resource>_IN_HAND
     - Number of `resource` cards in hand
     - 5
     - Integer
   * - P0_<dev-card>_IN_HAND
     - Number of `dev-card` cards in hand
     - 5
     - Integer
   * - P<i>_NUM_DEVS_IN_HAND
     - Number of hidden development cards player `i` has
     - N
     - Integer
   * - P<i>_NUM_RESOURCES _IN_HAND
     - Number of hidden resource cards player `i` has
     - N
     - Integer

   * - P<i>_HAS_ARMY
     - Whether player <i> has Largest Army
     - N
     - Boolean
   * - P<i>_HAS_ROAD
     - Whether player <i> has Longest Road
     - N
     - Boolean
   * - P<i>_ROADS_LEFT
     - Number of roads pieces player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_SETTLEMENTS_LEFT
     - Number of settlements player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_CITIES_LEFT
     - Number of cities player `i` has outside of board (left to build)
     - N
     - Integer
   * - P<i>_LONGEST_ROAD _LENGTH
     - Length of longest road by player `i`
     - N
     - Integer
   * - P<i>_PUBLIC_VPS
     - Amount of visible victory points for player `i` (i.e.
       doesn't include hidden victory point cards; only army,
       road and settlements/cities).
     - N
     - Integer
   * - P<i>_<dev-card>_PLAYED
     - Amount of `dev-card` cards player `i` has played in game
       (VICTORY_POINT not included).
     - 4 * N
     - Integer
   * - 
     - 
     - 194 * N + 226
     - 
"""

#
# def calc_clean_prod(state, p_color):
#     prob_dict = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
#     p0_total_payout = [0, 0, 0, 0, 0]
#     # p0_nodes = [list(), list(), list(), list(), list()]
#     for number, prob in prob_dict.items():
#         payout = calculate_resource_production_for_number(state.board, number)
#         if p_color in payout.keys():
#             p0_payout = payout[p_color]
#         else:
#             p0_payout = [0, 0, 0, 0, 0]
#
#         for i, amount in enumerate(p0_payout):
#             # for j in range(int(amount)):
#             #     p0_nodes[i].append(number)
#             p0_total_payout[i] += (amount * prob)
#     return p0_total_payout
#
# def calc_dev_card_in_hand(state, p_key):
#     dev_card_in_hand = 0
#     dev_card_in_hand += state.player_state[f"{p_key}_KNIGHT_IN_HAND"]
#     dev_card_in_hand += state.player_state[f"{p_key}_YEAR_OF_PLENTY_IN_HAND"]
#     dev_card_in_hand += state.player_state[f"{p_key}_ROAD_BUILDING_IN_HAND"]
#     dev_card_in_hand += state.player_state[f"{p_key}_MONOPOLY_IN_HAND"]
#     dev_card_in_hand += state.player_state[f"{p_key}_VICTORY_POINT_IN_HAND"]
#
#     return dev_card_in_hand
#
# def generate_x(game, p0_color):
#     p1_color = Color.RED
#     if p0_color == p1_color:
#         p1_color = Color.BLUE
#     p_key = player_key(game.state, p0_color)
#     p1_key = player_key(game.state, p1_color)
#
#     state = game.state
#     board = state.board
#     player_state = state.player_state
#     # # we will later add port features for each node
#     # X = np.zeros(648)
#     # for i in range(54):
#     #
#     #     if i in get_player_buildings(state, p0_color, SETTLEMENT):
#     #         X[12 * i] = 1
#     #     elif i in get_player_buildings(state, p1_color, SETTLEMENT):
#     #         X[12 * i] = -1
#     #
#     #     for j, resource in enumerate(RESOURCES):
#     #         entry_index = 12*i + j + 1
#     #         X[entry_index] = get_node_production(game.state.board.map, i, resource)
#     #
#     # return X
#
#
#     # p0_buildings = get_player_buildings(state, p0_color, SETTLEMENT)
#     # p1_buildings = get_player_buildings(state, p1_color, SETTLEMENT)
#     #
#     # X = np.zeros(324)
#     # for i in range(54):
#     #
#     #     if i in p0_buildings:
#     #         X[6 * i] = 1
#     #     elif i in p1_buildings:
#     #         X[6 * i] = -1
#     #
#     #     for j, resource in enumerate(RESOURCES):
#     #         X[(6 * i) + (j + 1)] = get_node_production(game.state.board.map, i, resource)
#     #
#     # return X
#
#     p0_settle = get_player_buildings(state, p0_color, SETTLEMENT)
#     p1_settle = get_player_buildings(state, p1_color, SETTLEMENT)
#     p0_city = get_player_buildings(state, p0_color, CITY)
#     p1_city = get_player_buildings(state, p1_color, CITY)
#
#     X = np.zeros(363)
#     for i in range(54):
#
#         if i in p0_settle:
#             X[6 * i] = 1
#         if i in p1_settle:
#             X[6 * i] = -1
#         if i in p0_city:
#             X[6 * i] = 2
#         if i in p1_city:
#             X[6 * i] = -2
#
#         for j, resource in enumerate(RESOURCES):
#             X[(6 * i) + (j + 1)] = get_node_production(game.state.board.map, i, resource)
#
#     # features for player 0 (BLUE / me)
#     X[324] = player_state[f"{p_key}_VICTORY_POINTS"]
#     X[325] = player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"]
#     X[326] = player_state[f"{p_key}_CITIES_AVAILABLE"]
#     X[327] = player_state[f"{p_key}_ROADS_AVAILABLE"] / 13
#     X[328] = player_state[f"{p_key}_PLAYED_KNIGHT"]
#     X[329] = player_state[f"{p_key}_HAS_ARMY"]
#     X[330] = player_state[f"{p_key}_HAS_ROAD"]
#     X[331] = player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
#     X[332] = calc_dev_card_in_hand(state, p_key)
#     for j, r in enumerate(RESOURCES):
#         X[333 + j] = player_state[f"{p_key}_{r}_IN_HAND"]
#     p0_prod = calc_clean_prod(game.state, p0_color)
#     for j, p in enumerate(p0_prod):
#         X[338+j] = p
#
#     # features for player 1 (RED / enemy)
#     X[343] = player_state[f"{p1_key}_VICTORY_POINTS"]
#     X[344] = player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"]
#     X[345] = player_state[f"{p1_key}_CITIES_AVAILABLE"]
#     X[346] = player_state[f"{p1_key}_ROADS_AVAILABLE"] / 13
#     X[347] = player_state[f"{p1_key}_PLAYED_KNIGHT"]
#     X[348] = player_state[f"{p1_key}_HAS_ARMY"]
#     X[349] = player_state[f"{p1_key}_HAS_ROAD"]
#     X[350] = player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
#     X[351] = calc_dev_card_in_hand(state, p1_key)
#     for j, r in enumerate(RESOURCES):
#         X[352 + j] = player_state[f"{p1_key}_{r}_IN_HAND"]
#     p1_prod = calc_clean_prod(game.state, p1_color)
#     for j, p in enumerate(p1_prod):
#         X[357 + j] = p
#
#     X[362] = state.num_turns
#
#     return X
#
# def get_node_production(catan_map, node_id, resource):
#
#     prob_dict = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
#     production = 0
#     for tile in catan_map.adjacent_tiles[node_id]:
#         if tile.resource == resource:
#             production += prob_dict[tile.number]
#
#     return production
#
#     # tiles = catan_map.adjacent_tiles[node_id]
#     # return sum([number_probability(t.number) for t in tiles if t.resource == resource])
