import random
from catanatron.game import Game
from catanatron import Player
# from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental.MichaelFiles.HeuristicFeatures import (simple_reward,
                                                                    calc_node_value,
                                                                    calc_tile_value_for_player,
                                                                    calc_road_value_for_player)
from catanatron.models.enums import ActionType
from catanatron_gym.envs.catanatron_env import from_action_space
from catanatron.models.player import Color
from catanatron_gym.features import get_player_expandable_nodes
from catanatron.state_functions import player_key

# @register_player("MYRB")
class MyRBPlayer(Player):
    def __init__(self, color, is_bot=True, epsilon=None):
        super().__init__(color, is_bot)
        self.epsilon = epsilon

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        #     # for exploration
        # if self.epsilon is not None and random.random() < self.epsilon:
        #     return random.choice(playable_actions)

        # p1_color = Color.RED
        # if self.color == p1_color:
        #     p1_color = Color.BLUE
        # p_key = player_key(game.state, self.color)
        # # p1_key = player_key(game.state, p1_color)

        best_value = float("-inf")
        possible_action_types = set()

        for action in playable_actions:
            possible_action_types.add(action.action_type)

        # rule_based_action = playable_actions[0]
        if ActionType.BUILD_SETTLEMENT in possible_action_types:
            return self.build_settlement(game, playable_actions)
        elif ActionType.BUILD_CITY in possible_action_types:
            return self.build_city(game, playable_actions)
        elif ActionType.BUILD_ROAD in possible_action_types:
            road_val, road_action = self.build_road(game, playable_actions)
            # if road_val > 2:
            #     return road_action
            return road_action

        elif ActionType.MOVE_ROBBER in possible_action_types:
            return self.move_robber(game, playable_actions)

        # player_state = game.state.player_state
        # resources = {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}
        # total_cards_in_hand = 0
        # for r in resources:
        #     total_cards_in_hand += player_state[f"{p_key}_{r}_IN_HAND"]

        best_actions = []
        for action in playable_actions:
            if ActionType.BUY_DEVELOPMENT_CARD == action.action_type:
                continue

            game_copy = game.copy()
            if isinstance(action, int):
                catan_action = from_action_space(action, game.state.playable_actions)
                game_copy.execute(catan_action)
            else:
                game_copy.execute(action)

            value = simple_reward(game_copy, self.color)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_value = value
                best_actions = [action]

        chosen_action = random.choice(best_actions)
        # print(f"rule_based_action = {rule_based_action.action_type} , {rule_based_action.value}")
        # print(f"chosen_action = {chosen_action.action_type} , {chosen_action.value}")
        return chosen_action # from all equally best actions randomly pick 1

    def build_settlement(self, game, playable_actions):
        node_ids = dict()
        for action in playable_actions:
            if ActionType.BUILD_SETTLEMENT == action.action_type:
                node_ids[action.value] = action

        best_value = float("-inf")
        best_node = 0
        for node_id, action in node_ids.items():
            node_value = calc_node_value(game, node_id, self.color)
            if node_value > best_value:
                best_value = node_value
                best_node = node_id

        return node_ids[best_node]

    def build_city(self, game, playable_actions):
        node_ids = dict()
        for action in playable_actions:
            if ActionType.BUILD_CITY == action.action_type:
                node_ids[action.value] = action

        best_value = float("-inf")
        best_node = 0
        for node_id, action in node_ids.items():
            node_value = calc_node_value(game, node_id, self.color)
            if node_value > best_value:
                best_value = node_value
                best_node = node_id

        return node_ids[best_node]

    def build_road(self, game, playable_actions):

        p1_color = Color.RED
        if p1_color == self.color:
            p1_color = Color.BLUE

        node_ids = dict()
        for action in playable_actions:
            if ActionType.BUILD_ROAD == action.action_type:
                node_ids[action.value] = action

        best_value = float("-inf")
        best_pair = list(node_ids.keys())[0]
        for node_pair, action in node_ids.items():
            node_value = calc_road_value_for_player(game, node_pair, self.color)
            if node_value > best_value:
                best_value = node_value
                best_pair = node_pair

        return best_value, node_ids[best_pair]

    def move_robber(self, game, playable_actions):
        board = game.state.board

        p1_color = Color.RED
        if p1_color == self.color:
            p1_color = Color.BLUE

        hex_ids = dict()
        for action in playable_actions:
            if ActionType.MOVE_ROBBER == action.action_type:
                hex_ids[(action.value)[0]] = action

        best_value = float("-inf")
        best_hex = list(hex_ids.keys())[0]
        for hex_id, action in hex_ids.items():
            tile = game.state.board.map.land_tiles[hex_id]
            if tile.resource is None:
                continue
            node_value = calc_tile_value_for_player(game, hex_id, p1_color)
            if node_value > best_value:
                best_value = node_value
                best_hex = hex_id

        return hex_ids[best_hex]

