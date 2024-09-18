
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, FastResource
from catanatron.state_functions import get_player_buildings
from catanatron.models.decks import freqdeck_replenish
from catanatron.state_functions import player_key
from catanatron.models.player import Color
from collections import defaultdict
from catanatron.game import Game
from typing import List, Dict
import numpy as np
from catanatron_gym.features import get_player_expandable_nodes
from catanatron.models.board import get_node_distances

def calculate_resource_production_for_number(board, number):
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
            # p0_total_payout[i] += (((amount+1)/2) * prob)  # 4/36 prob to get 1 is better than 2/36 to get 2
            p0_total_payout[i] += (amount * prob)
            if i==4:    #ore
                p0_total_payout[i] += 0.1

    production_value = 0

    for p_val in p0_total_payout:
        if p_val <= 0:
            production_value -= 3   # penalty for lack of resource diversity
        else:
            production_value += p_val
    return production_value


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

    p0_settle = get_player_buildings(state, p0_color, SETTLEMENT)
    p1_settle = get_player_buildings(state, p1_color, SETTLEMENT)
    p0_city = get_player_buildings(state, p0_color, CITY)
    p1_city = get_player_buildings(state, p1_color, CITY)

    p0_expandable_nodes = get_player_expandable_nodes(game, p0_color)
    p1_expandable_nodes = get_player_expandable_nodes(game, p1_color)

    distances = get_node_distances()

    p0_port_resources = board.get_player_port_resources(p0_color)
    p1_port_resources = board.get_player_port_resources(p1_color)

    X = np.zeros(477)
    for i in range(54):

        if i in p0_settle:
            X[8 * i] = 1
        if i in p1_settle:
            X[8 * i] = -1
        if i in p0_city:
            X[8 * i] = 2
        if i in p1_city:
            X[8 * i] = -2

        for j, resource in enumerate(RESOURCES):
            X[(8 * i) + (j + 1)] = get_node_production(game.state.board, i, resource)

        if len(p0_expandable_nodes) > 0:
            p0_min_distance_to_node_i = min([distances[i][p0_node] for p0_node in p0_expandable_nodes])
        else:
            p0_min_distance_to_node_i = 15
        X[(8 * i) + 6] = p0_min_distance_to_node_i
        if len(p1_expandable_nodes) > 0:
            p1_min_distance_to_node_i = min([distances[i][p1_node] for p1_node in p1_expandable_nodes])
        else:
            p1_min_distance_to_node_i = 15
        # p1_min_distance_to_node_i = min([distances[i][p1_node] for p1_node in p1_expandable_nodes])
        X[(8 * i) + 7] = p1_min_distance_to_node_i

    # features for player 0 (BLUE / me)
    X[432] = calc_dev_card_in_hand(state, p_key)
    X[433] = player_state[f"{p_key}_ROADS_AVAILABLE"] / 13
    X[434] = player_state[f"{p_key}_PLAYED_KNIGHT"]
    X[435] = player_state[f"{p_key}_HAS_ARMY"]
    X[436] = player_state[f"{p_key}_HAS_ROAD"]
    X[437] = player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    for j, r in enumerate(RESOURCES):
        X[438 + j] = player_state[f"{p_key}_{r}_IN_HAND"]
    p0_prod = calc_clean_prod(game.state, p0_color)
    for j, p in enumerate(p0_prod):
        X[443+j] = p
    if None in p0_port_resources:   # if has 3:1 port
        X[448] = 1
    for j, r in enumerate(RESOURCES):   # if has 2:1 port
        if r in p0_port_resources:
            X[449 + j] = 1


    # features for player 1 (RED / enemy)
    X[454] = calc_dev_card_in_hand(state, p1_key)
    X[455] = player_state[f"{p1_key}_ROADS_AVAILABLE"] / 13
    X[456] = player_state[f"{p1_key}_PLAYED_KNIGHT"]
    X[457] = player_state[f"{p1_key}_HAS_ARMY"]
    X[458] = player_state[f"{p1_key}_HAS_ROAD"]
    X[459] = player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
    for j, r in enumerate(RESOURCES):
        X[460 + j] = player_state[f"{p1_key}_{r}_IN_HAND"]
    p1_prod = calc_clean_prod(game.state, p1_color)
    for j, p in enumerate(p1_prod):
        X[465 + j] = p
    if None in p1_port_resources:   # if has 3:1 port
        X[470] = 1
    for j, r in enumerate(RESOURCES):   # if has 2:1 port
        if r in p1_port_resources:
            X[471 + j] = 1

    X[476] = state.num_turns

    return X

def get_node_production(board, node_id, resource):

    catan_map = board.map
    robber_coordinates = board.robber_coordinate
    robber_tile = board.map.land_tiles[robber_coordinates]  # gives penalty to the robber tile
    robber_resource = robber_tile.resource
    robber_nodes = robber_tile.nodes.values()

    prob_dict = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    production = 0
    for tile in catan_map.adjacent_tiles[node_id]:
        robber_penalty = 1
        if tile.id == robber_tile.id:
            robber_penalty = 0.1
        if tile.resource == resource:
            production += prob_dict[tile.number] * robber_penalty

    return production





