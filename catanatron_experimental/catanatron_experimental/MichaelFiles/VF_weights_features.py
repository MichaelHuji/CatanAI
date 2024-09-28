
from catanatron.models.enums import RESOURCES
from catanatron.state_functions import player_key
from catanatron.models.player import Color
import numpy as np
from catanatron_gym.features import get_player_expandable_nodes
from catanatron.models.board import get_node_distances
from catanatron_experimental.MichaelFiles.Features import get_node_production, calc_clean_prod, calc_dev_card_in_hand

def generate_feature_vector(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    state = game.state
    board = state.board
    player_state = state.player_state

    X = np.zeros(91)

    # features for player 0 (BLUE / me)
    X[0] = player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"]
    X[1] = player_state[f"{p_key}_VICTORY_POINTS"]
    X[2] = 5 - player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"] # number of settlements on board
    X[3] = 4 - player_state[f"{p_key}_CITIES_AVAILABLE"] # number of cities on board
    X[4] = 13 - player_state[f"{p_key}_ROADS_AVAILABLE"]
    X[5] = player_state[f"{p_key}_PLAYED_KNIGHT"]
    X[6] = player_state[f"{p_key}_HAS_ARMY"]
    X[7] = player_state[f"{p_key}_HAS_ROAD"]
    X[8] = player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    resources_in_hand = 0
    for j, r in enumerate(RESOURCES):
        X[9 + j] = player_state[f"{p_key}_{r}_IN_HAND"]
        resources_in_hand += X[9 + j]
    X[14] = resources_in_hand
    p0_prod = calc_clean_prod(game.state, p0_color)
    num_of_producable_resources = 0
    for j, p in enumerate(p0_prod):
        X[15+j] = p
        if p > 0:
            num_of_producable_resources += 1
    X[20] = calc_dev_card_in_hand(state, p_key)

    # distance from longest road
    if max(player_state[f"{p_key}_LONGEST_ROAD_LENGTH"], player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]) < 5:
        X[21] = 5 - player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    else:
        X[21] = player_state[f"{p_key}_LONGEST_ROAD_LENGTH"] - player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]

    # distance from largest army
    if max(player_state[f"{p_key}_PLAYED_KNIGHT"], player_state[f"{p1_key}_PLAYED_KNIGHT"]) < 3:
        X[22] = 3 - player_state[f"{p_key}_PLAYED_KNIGHT"]
    else:
        X[22] = player_state[f"{p_key}_PLAYED_KNIGHT"] - player_state[f"{p1_key}_PLAYED_KNIGHT"]

    p0_port_resources = board.get_player_port_resources(p0_color)
    if None in p0_port_resources:
        X[23] = 1
    for i, resource in enumerate(RESOURCES):
        if resource in p0_port_resources:
            X[24+i] = 1

    X[29] = num_of_producable_resources

    p0_expandable_nodes = get_player_expandable_nodes(game, p0_color)
    p1_expandable_nodes = get_player_expandable_nodes(game, p1_color)
    X[30] = len(p0_expandable_nodes)

    p0_buildable_node_ids = sorted(list(board.board_buildable_ids))
    X[31] = 0   # deleted a feature, make it compatible

    distances = get_node_distances()
    best_dist_0 = -1
    best_dist_1 = -1
    best_0_dist_from_enemy = -1
    best_1_dist_from_enemy = -1

    for p0_buildable_node_id in p0_buildable_node_ids:
        if len(p0_expandable_nodes) <= 0:
            break
        p0_min_distance_to_node = min([distances[p0_buildable_node_id][p0_node] for p0_node in p0_expandable_nodes])
        if p0_min_distance_to_node > 1:
            continue
        road_value = 0
        for resource in RESOURCES:
            road_value += get_node_production(game.state.board.map, p0_buildable_node_id, resource)

        if road_value  > best_dist_1:
            best_dist_1 = road_value
            best_1_dist_from_enemy = 10
            if len(p1_expandable_nodes) > 0:
                best_1_dist_from_enemy = min([distances[p0_buildable_node_id][p1_node] for p1_node in p1_expandable_nodes])
        if p0_min_distance_to_node == 0 and road_value > best_dist_0:
            best_dist_0 = road_value
            best_0_dist_from_enemy = 10
            if len(p1_expandable_nodes) > 0:
                best_0_dist_from_enemy = min([distances[p0_buildable_node_id][p1_node] for p1_node in p1_expandable_nodes])

    X[32] = best_dist_1
    X[33] = best_dist_0
    X[34] = best_1_dist_from_enemy
    X[35] = best_0_dist_from_enemy

    missing_for_settlement = 4
    for r in ["WOOD", "BRICK", "SHEEP", "WHEAT"]:
        if player_state[f"{p_key}_{r}_IN_HAND"] > 0:
            missing_for_settlement -= 1
    X[36] = missing_for_settlement

    missing_for_city = 5 - (min(2, player_state[f"{p_key}_WHEAT_IN_HAND"]) + min(3, player_state[f"{p_key}_WHEAT_IN_HAND"]))
    X[37] = missing_for_city

    # features for player 1 (RED / enemy)
    X[38] = player_state[f"{p1_key}_VICTORY_POINTS"]
    X[39] = 5 - player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"]     # number of settlements on board
    X[40] = 4 - player_state[f"{p1_key}_CITIES_AVAILABLE"]     # number of cities on board
    X[41] = 13 - player_state[f"{p1_key}_ROADS_AVAILABLE"]
    X[42] = player_state[f"{p1_key}_PLAYED_KNIGHT"]
    X[43] = player_state[f"{p1_key}_HAS_ARMY"]
    X[44] = player_state[f"{p1_key}_HAS_ROAD"]
    X[45] = player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]

    resources_in_hand = 0
    for j, r in enumerate(RESOURCES):
        X[46 + j] = player_state[f"{p1_key}_{r}_IN_HAND"]
        resources_in_hand += X[9 + j]
    X[51] = resources_in_hand

    p1_prod = calc_clean_prod(game.state, p1_color)
    p1_num_of_producable_resources = 0
    for j, p in enumerate(p1_prod):
        X[52+j] = p
        if p > 0:
            p1_num_of_producable_resources += 1

    X[57] = calc_dev_card_in_hand(state, p1_key)

    # distance from longest road
    if max(player_state[f"{p_key}_LONGEST_ROAD_LENGTH"], player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]) < 5:
        X[58] = 5 - player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
    else:
        X[58] = player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"] - player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]

    # distance from largest army
    if max(player_state[f"{p_key}_PLAYED_KNIGHT"], player_state[f"{p1_key}_PLAYED_KNIGHT"]) < 3:
        X[59] = 3 - player_state[f"{p1_key}_PLAYED_KNIGHT"]
    else:
        X[59] = player_state[f"{p1_key}_PLAYED_KNIGHT"] - player_state[f"{p_key}_PLAYED_KNIGHT"]

    p1_port_resources = board.get_player_port_resources(p1_color)
    if None in p1_port_resources:
        X[60] = 1
    for i, resource in enumerate(RESOURCES):
        if resource in p1_port_resources:
            X[61+i] = 1

    X[66] = p1_num_of_producable_resources
    X[67] = len(p1_expandable_nodes)

    best_dist_0_1 = -1
    best_dist_1_1 = -1
    best_0_dist_from_me = -1
    best_1_dist_from_me = -1

    for p0_buildable_node_id in p0_buildable_node_ids:
        if len(p1_expandable_nodes) <= 0:
            break
        p1_min_distance_to_node = min([distances[p0_buildable_node_id][p1_node] for p1_node in p1_expandable_nodes])
        if p1_min_distance_to_node > 1:
            continue
        road_value = 0
        for resource in RESOURCES:
            road_value += get_node_production(game.state.board.map, p0_buildable_node_id, resource)

        if road_value > best_dist_1_1:
            best_dist_1_1 = road_value
            best_1_dist_from_me = 0
            if len(p0_expandable_nodes) > 0:
                best_1_dist_from_me = min([distances[p0_buildable_node_id][p0_node] for p0_node in p0_expandable_nodes])
        if p1_min_distance_to_node == 0 and road_value > best_dist_0_1:
            best_dist_0_1 = road_value
            best_0_dist_from_me = 0
            if len(p0_expandable_nodes) > 0:
                best_0_dist_from_me = min([distances[p0_buildable_node_id][p0_node] for p0_node in p0_expandable_nodes])

    X[68] = best_dist_1_1
    X[69] = best_dist_0_1
    X[70] = best_1_dist_from_me
    X[71] = best_0_dist_from_me

    missing_for_settlement = 4
    for r in ["WOOD", "BRICK", "SHEEP", "WHEAT"]:
        if player_state[f"{p1_key}_{r}_IN_HAND"] > 0:
            missing_for_settlement -= 1
    X[72] = missing_for_settlement

    missing_for_city = 5 - (min(2, player_state[f"{p1_key}_WHEAT_IN_HAND"]) + min(3, player_state[f"{p1_key}_WHEAT_IN_HAND"]))
    X[73] = missing_for_city

    X[74] = 0

    X[75] = X[2] + X[3]
    X[76] = np.abs(X[15] - X[16])
    X[77] = np.abs((X[18] / 2) - (X[19] / 3))
    X[78] = X[15] * X[24]
    X[79] = X[16] * X[25]
    X[80] = X[17] * X[26]
    X[81] = X[18] * X[27]
    X[82] = X[19] * X[28]

    X[83] = X[39] + X[40]
    X[84] = np.abs(X[52] - X[53])
    X[85] = np.abs((X[55] / 2) - (X[56] / 3))
    X[86] = X[52] * X[61]
    X[87] = X[53] * X[62]
    X[88] = X[54] * X[63]
    X[89] = X[55] * X[64]
    X[90] = X[56] * X[65]

    return X





