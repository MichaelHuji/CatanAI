
from catanatron.models.board import get_node_distances
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, FastResource
from catanatron.state_functions import get_player_buildings
from catanatron.models.decks import freqdeck_replenish
from catanatron.state_functions import player_key
from catanatron.models.player import Color
from collections import defaultdict
from catanatron.game import Game
from typing import List, Dict
import numpy as np
from catanatron_experimental.MichaelFiles.Features import (calculate_resource_production_for_number,
                                                           calculate_resource_production_value_for_player)
from catanatron_gym.features import get_player_expandable_nodes
from catanatron_experimental.MichaelFiles.Features import get_node_production


def calc_road_building_direction_reward(game, p0_color):

    state = game.state
    board = state.board
    p0_expandable_nodes = get_player_expandable_nodes(game, p0_color)
    distances = get_node_distances()

    p0_buildable_node_ids = sorted(list(board.board_buildable_ids))

    best_buildable_node_value = -1
    for p0_buildable_node_id in p0_buildable_node_ids:
        if len(p0_expandable_nodes) <= 0:
            break
        p0_min_distance_to_node = min([distances[p0_buildable_node_id][p0_node] for p0_node in p0_expandable_nodes])
        # print(p0_min_distance_to_node)
        if p0_min_distance_to_node > 1:
            continue
        road_value = 0
        reward_factor = 1
        if p0_min_distance_to_node == 0:
            reward_factor = 1.5
        for resource in RESOURCES:
            road_value += get_node_production(game.state.board.map, p0_buildable_node_id, resource)
        if road_value * reward_factor > best_buildable_node_value:
            best_buildable_node_value = road_value * reward_factor

    return best_buildable_node_value

def calc_port_reward(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    state = game.state
    board = state.board
    port_reward = 0
    p0_port_resources = board.get_player_port_resources(p0_color)
    if None in p0_port_resources:
        port_reward = 1

    for resource in RESOURCES:
        if resource in p0_port_resources:
            port_reward += 0.5

    return port_reward

def initial_stage_reward(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    production_reward = calc_init_production_val(game.state, p0_color)
    production_reward -= calc_init_production_val(game.state, p1_color)

    resource_reward = calc_resource_reward(game.state, p_key, p0_color)
    resource_reward -= calc_resource_reward(game.state, p1_key, p1_color)

    expand_reward = 0
    expand_reward += calc_road_building_direction_reward(game, p0_color) / 10
    expand_reward -= calc_road_building_direction_reward(game, p1_color) / 10
    port_reward = calc_port_reward(game, p0_color) / 10
    total_reward = production_reward + resource_reward + expand_reward + port_reward
    return total_reward

def calc_init_production_val(state, p0_color):
    # the probabilities for 6, 8 are not 5 because they have a higher chance to get blocked
    prob_dict = {2: 1, 3: 1.95, 4: 2.85, 5: 3.7, 6: 4.5, 7: 0, 8: 4.5, 9: 3.7, 10: 2.85, 11: 1.95, 12: 1}
    p0_total_payout = [0, 0, 0, 0, 0]
    for number, prob in prob_dict.items():
        payout = calculate_resource_production_for_number(state.board, number)
        if p0_color in payout.keys():
            p0_payout = payout[p0_color]
        else:
            p0_payout = [0, 0, 0, 0, 0]
        for i, amount in enumerate(p0_payout):
            if amount == 0:
                continue
            p0_total_payout[i] += (amount * prob)
            if i == 4:
                p0_total_payout[i] -= 0.02
            if i == 2:
                p0_total_payout[i] -= 0.01

    production_value = 0
    for p_val in p0_total_payout:
        if p_val <= 0:
            production_value -= 3.6  # penalty for lack of resource diversity
        else:
            production_value += p_val
    return production_value


def end_stage_reward(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    winning_reward = get_winning_reward(game, p0_color)
    if winning_reward != 0:
        return winning_reward

    # return 0
    total_reward = 0
    production_reward = 0
    production_reward += calculate_resource_production_value_for_player(game.state, p0_color, 0.9)
    production_reward -= calculate_resource_production_value_for_player(game.state, p1_color, 0.7)
    production_reward *= 0.5

    settlement_reward = 2 * (5 - game.state.player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"])
    settlement_reward -= 2 * (5 - game.state.player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"])

    city_reward = 3 * (4 - game.state.player_state[f"{p_key}_CITIES_AVAILABLE"])
    city_reward -= 3 * (4 - game.state.player_state[f"{p1_key}_CITIES_AVAILABLE"])

    road_reward = game.state.player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    road_reward -= game.state.player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
    if game.state.player_state[f"{p_key}_HAS_ROAD"]:
        road_reward += 10
    elif game.state.player_state[f"{p1_key}_HAS_ROAD"]:
        road_reward -= 10

    resource_reward = calc_resource_reward(game.state, p_key, p0_color)
    resource_reward -= calc_resource_reward(game.state, p1_key, p1_color)

    port_reward = calc_port_reward(game, p0_color) / 10

    total_reward += production_reward
    total_reward += settlement_reward
    total_reward += city_reward
    total_reward += road_reward
    total_reward += resource_reward
    total_reward += port_reward
    return total_reward

# function to give reward for the current state of the game
def simple_reward(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    if game.state.player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"] < 3:
        return initial_stage_reward(game, p0_color)

    num_nodes = (5 - game.state.player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"])
    num_nodes += 2 * (4 - game.state.player_state[f"{p_key}_CITIES_AVAILABLE"])

    p_1_num_nodes = (5 - game.state.player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"])
    p_1_num_nodes += 2 * (4 - game.state.player_state[f"{p1_key}_CITIES_AVAILABLE"])
    if num_nodes >= 6 or p_1_num_nodes >= 8:
        return end_stage_reward(game, p0_color)

    total_reward = 0
    production_reward = 0
    production_reward += calculate_resource_production_value_for_player(game.state, p0_color)
    production_reward -= calculate_resource_production_value_for_player(game.state, p1_color)

    settlement_reward = 2 * (5 - game.state.player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"])
    settlement_reward -= 2 * (5 - game.state.player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"])

    city_reward = 3 * (4 - game.state.player_state[f"{p_key}_CITIES_AVAILABLE"])
    city_reward -= 3 * (4 - game.state.player_state[f"{p1_key}_CITIES_AVAILABLE"])

    road_reward = 0.3 * game.state.player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    road_reward -= 0.3 * game.state.player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
    if game.state.player_state[f"{p_key}_HAS_ROAD"]:
        road_reward += 3
    elif game.state.player_state[f"{p1_key}_HAS_ROAD"]:
        road_reward -= 3

    expand_reward = 0
    expand_reward += calc_road_building_direction_reward(game, p0_color) / 5
    development_card_reward = calc_development_card_reward(game.state, p_key)
    resource_reward = calc_resource_reward(game.state, p_key, p0_color)
    resource_reward -= calc_resource_reward(game.state, p1_key, p1_color)

    port_reward = calc_port_reward(game, p0_color) / 10

    total_reward += production_reward
    total_reward += settlement_reward
    total_reward += city_reward
    total_reward += road_reward
    total_reward += development_card_reward
    total_reward += resource_reward
    total_reward += expand_reward
    total_reward += port_reward
    return total_reward

def get_winning_reward(game, p0_color):
    winning_color = game.winning_color()
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    if p0_color == winning_color:
        return 1000
    elif winning_color is None:
        return 0
    else:
        return -1000

def can_build_settlement(state, color):
    key = player_key(state, color)

    if state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] > 0:
        buildable_node_ids = state.board.buildable_node_ids(color)
        return len(buildable_node_ids)
    else:
        return 0

def has_settlement_to_upgrade(state, color):
    key = player_key(state, color)

    has_cities_available = state.player_state[f"{key}_CITIES_AVAILABLE"] > 0
    if not has_cities_available:
        return 0

    return len(get_player_buildings(state, color, SETTLEMENT))

def calc_development_card_reward(state, p_key):
    dev_card_reward = 0
    # dev_card_reward += 0.15 * state.player_state[f"{p_key}_KNIGHT_IN_HAND"]
    # dev_card_reward += 0.2 * state.player_state[f"{p_key}_YEAR_OF_PLENTY_IN_HAND"]
    # dev_card_reward += 0.25 * state.player_state[f"{p_key}_ROAD_BUILDING_IN_HAND"]
    # dev_card_reward += 0.3 * state.player_state[f"{p_key}_MONOPOLY_IN_HAND"]
    # dev_card_reward += 0.3 * state.player_state[f"{p_key}_VICTORY_POINT_IN_HAND"]

    dev_card_reward -= 0.1 * state.player_state[f"{p_key}_KNIGHT_IN_HAND"]
    dev_card_reward -= 0.1 * state.player_state[f"{p_key}_YEAR_OF_PLENTY_IN_HAND"]
    dev_card_reward -= 0.1 * state.player_state[f"{p_key}_ROAD_BUILDING_IN_HAND"]
    dev_card_reward -= 0.1 * state.player_state[f"{p_key}_MONOPOLY_IN_HAND"]
    dev_card_reward -= 0.1 * state.player_state[f"{p_key}_VICTORY_POINT_IN_HAND"]

    return dev_card_reward


def calc_resource_reward(state, p_key, color):
    resource_reward = 0
    resources = {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}

    total_cards_in_hand = 0
    for r in resources:
        total_cards_in_hand += 0.1 * state.player_state[f"{p_key}_{r}_IN_HAND"]

    settlement_locations = can_build_settlement(state, color)
    if settlement_locations > 0:
        resource_reward += 0.45       # we want to have locations to build so we reward
        resource_reward += 0.05*settlement_locations
        miss_for_set = calc_missing_resources_for_settlement(state, p_key)
        if miss_for_set == 0:
            resource_reward += 0.3
        elif miss_for_set == 1:
            resource_reward += 0.2

    if has_settlement_to_upgrade(state, color) > 0:
        miss_for_city = calc_missing_resources_for_city(state, p_key)
        if miss_for_city == 0:
            resource_reward += 1
        elif miss_for_city == 1:
            resource_reward += 0.8
        elif miss_for_city == 2:
            resource_reward += 0.6
        elif miss_for_city == 3:
            resource_reward += 0.2

    if total_cards_in_hand > 7:
        resource_reward -= 0.1
    return resource_reward


def endgame_resource_reward(state, p_key, color):
    resource_reward = 0
    resources = {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}
    total_cards_in_hand = 0
    wood_in_hand = state.player_state[f"{p_key}_WOOD_IN_HAND"]
    total_cards_in_hand += wood_in_hand
    resource_reward += wood_in_hand

    brick_in_hand = state.player_state[f"{p_key}_BRICK_IN_HAND"]
    total_cards_in_hand += brick_in_hand
    resource_reward += brick_in_hand

    sheep_in_hand = state.player_state[f"{p_key}_SHEEP_IN_HAND"]
    total_cards_in_hand += sheep_in_hand
    resource_reward += sheep_in_hand

    wheat_in_hand = state.player_state[f"{p_key}_WHEAT_IN_HAND"]
    total_cards_in_hand += wheat_in_hand
    if wheat_in_hand == 1:
        resource_reward += 3
    elif wheat_in_hand == 2:
        resource_reward += 6
    elif wheat_in_hand > 2:
        resource_reward += 6
        resource_reward += wheat_in_hand - 2

    ore_in_hand = state.player_state[f"{p_key}_ORE_IN_HAND"]
    total_cards_in_hand += ore_in_hand
    if ore_in_hand == 1:
        resource_reward += 3
    elif ore_in_hand == 2:
        resource_reward += 6
    elif ore_in_hand == 3:
        resource_reward += 9
    elif ore_in_hand > 3:
        resource_reward += 9
        resource_reward += ore_in_hand - 2

    settlement_locations = can_build_settlement(state, color)
    if settlement_locations > 0:
        resource_reward += 15       # we want to have locations to build so we reward
        resource_reward += 5*settlement_locations
        miss_for_set = calc_missing_resources_for_settlement(state, p_key)
        if miss_for_set == 0:
            resource_reward += 15
        elif miss_for_set == 1:
            resource_reward += 10

    if has_settlement_to_upgrade(state, color) > 0:
        miss_for_city = calc_missing_resources_for_city(state, p_key)
        if miss_for_city == 0:
            resource_reward += 25
        elif miss_for_city == 1:
            resource_reward += 15

    if total_cards_in_hand > 7:
        resource_reward -= 3

    return (resource_reward / 4) - 5

def calc_missing_resources_for_settlement(state, p_key):
    missing_cards = 4
    if state.player_state[f"{p_key}_WOOD_IN_HAND"] > 0:
        missing_cards -= 1
    if state.player_state[f"{p_key}_BRICK_IN_HAND"] > 0:
        missing_cards -= 1
    if state.player_state[f"{p_key}_SHEEP_IN_HAND"] > 0:
        missing_cards -= 1
    if state.player_state[f"{p_key}_WHEAT_IN_HAND"] > 0:
        missing_cards -= 1

    return missing_cards

def calc_missing_resources_for_city(state, p_key):
    missing_cards = 5
    ore_in_hand = state.player_state[f"{p_key}_ORE_IN_HAND"]
    if ore_in_hand > 2:
        missing_cards -= 3
    else:
        missing_cards -= ore_in_hand

    wheat_in_hand = state.player_state[f"{p_key}_WHEAT_IN_HAND"]
    if wheat_in_hand > 1:
        missing_cards -= 2
    else:
        missing_cards -= wheat_in_hand

    return missing_cards

def calc_node_value(game, node_id, p0_color):

    board = game.state.board
    catan_map = board.map

    # the probabilities for 6, 8 are not 5 because they have a higher chance to get blocked
    prob_dict = {2: 1, 3: 1.95, 4: 2.85, 5: 3.7, 6: 4.5, 7: 0, 8: 4.5, 9: 3.7, 10: 2.85, 11: 1.95, 12: 1}
    resource_dict = {"WOOD": 0, "BRICK": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
    p0_total_payout = [0, 0, 0, 0, 0]

    for number, prob in prob_dict.items():
        payout = calculate_resource_production_for_number(board, number)
        if p0_color in payout.keys():
            p0_payout = payout[p0_color]
        else:
            p0_payout = [0, 0, 0, 0, 0]
        for i, amount in enumerate(p0_payout):
            if amount == 0:
                continue
            p0_total_payout[i] += (amount * prob)

    for tile in catan_map.adjacent_tiles[node_id]:
        p0_total_payout[resource_dict.get(tile.resource, 0)] += prob_dict.get(tile.number, 0)

    production_value = 0
    for p_val in p0_total_payout:
        if p_val <= 0:
            production_value -= 3.6  # penalty for lack of resource diversity
        else:
            production_value += p_val

    return production_value


def calc_road_value_for_player(game, node_pair, p0_color):
    p0_expandable_nodes = get_player_expandable_nodes(game, p0_color)
    board = game.state.board
    board_buildable_ids = board.board_buildable_ids
    distances = get_node_distances()
    if (node_pair[0] in p0_expandable_nodes) and (node_pair[1] in p0_expandable_nodes):
        # we connect 2 roads, it's not that good, unless we want to get the longest path
        return 1

    elif (node_pair[0] in p0_expandable_nodes) and (node_pair[1] not in p0_expandable_nodes):
        # we build to a new location node_pair[1]
        # check if we can build in this new location

        if node_pair[1] in board_buildable_ids:   # if we can: check the value of that location
            return 20 + calc_node_value(game, node_pair[1], p0_color)
        else:   # if we can't, check if we can build in any of his neighbors
            value = 0
            for buildable_node in board_buildable_ids:  # if we can, check the value of the neighbors
                if buildable_node not in p0_expandable_nodes and distances[buildable_node][node_pair[1]] <= 1.5:
                    value = max(value, calc_node_value(game, buildable_node, p0_color))
            return value

    elif (node_pair[0] not in p0_expandable_nodes) and (node_pair[1] in p0_expandable_nodes):
        # check if we can build in this new location
        if node_pair[0] in board_buildable_ids:  # if we can: check the value of that location
            return 20 + calc_node_value(game, node_pair[1], p0_color)
        else:  # if we can't, check if we can build in any of his neighbors
            value = 0
            for buildable_node in board_buildable_ids:  # if we can, check the value of the neighbors
                if buildable_node not in p0_expandable_nodes and distances[buildable_node][node_pair[0]] <= 1.5:
                    value = max(value, calc_node_value(game, buildable_node, p0_color))
            return value
    return 1

def calc_tile_value_for_player(game, tile_id, p1_color):
    hex = game.state.board.map.land_tiles[tile_id]
    prob_dict = {2: 0.5, 3: 1, 4: 2, 5: 3.5, 6: 5, 7: 0, 8: 5, 9: 3.5, 10: 2, 11: 1, 12: 0.5}
    board = game.state.board
    tile_number = hex.number
    tile_value = 0
    for node_id in hex.nodes.values():
        building = board.buildings.get(node_id, None)
        gain = 0
        if building is None:
            continue
        elif building[1] == SETTLEMENT:
            gain = 1
        elif building[1] == CITY:   # city is worth more
            gain = 2
        if building[0] != p1_color: # if building is mine, give penalty
            gain *= -2
        tile_value += gain * (prob_dict[tile_number])

    return tile_value
