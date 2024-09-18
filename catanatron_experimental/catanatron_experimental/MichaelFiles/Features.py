
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, FastResource
from catanatron.state_functions import get_player_buildings
from catanatron.models.decks import freqdeck_replenish
from catanatron.state_functions import player_key
from catanatron.models.player import Color
from collections import defaultdict
from catanatron.game import Game
from typing import List, Dict
import numpy as np


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




