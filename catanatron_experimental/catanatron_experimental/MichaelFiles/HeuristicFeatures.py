
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

    return production_reward + resource_reward

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
            if i == 1:
                p0_total_payout[i] += 0.18

    production_value = 0
    for p_val in p0_total_payout:
        if p_val <= 0:
            production_value -= 3.6  # penalty for lack of resource diversity
        else:
            production_value += p_val
    return production_value

def calc_clean_prod(state, p_color):
    prob_dict = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    p0_total_payout = [0, 0, 0, 0, 0]
    p0_nodes = [list(), list(), list(), list(), list()]
    for number, prob in prob_dict.items():
        payout = calculate_resource_production_for_number(state.board, number)
        if p_color in payout.keys():
            p0_payout = payout[p_color]
        else:
            p0_payout = [0, 0, 0, 0, 0]

        for i, amount in enumerate(p0_payout):
            for j in range(int(amount)):
                p0_nodes[i].append(number)
            # p0_total_payout[i] += (amount * prob)
    return p0_nodes

def initial_stage_reward2(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    production_reward = calc_init_production_val2(game.state, p0_color)
    production_reward -= calc_init_production_val2(game.state, p1_color)

    resource_reward = calc_resource_reward(game.state, p_key, p0_color)
    resource_reward -= calc_resource_reward(game.state, p1_key, p1_color)

    return production_reward + resource_reward

def calc_init_production_val2(state, p0_color):
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

    production_value = 0
    for p_val in p0_total_payout:
        if p_val <= 0:
            production_value -= 3.6   # penalty for lack of resource diversity
        else:
            production_value += p_val
    return production_value

def end_stage_reward(game, p0_color):
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, Color.RED)
    total_reward = 0


    winning_reward = get_winning_reward(game, p0_color)
    if winning_reward != 0:
        return winning_reward
    settlement_reward = 5 - game.state.player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"]
    settlement_reward -= (5 - game.state.player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"])
    settlement_reward *= 50

    city_reward = 4 - game.state.player_state[f"{p_key}_CITIES_AVAILABLE"]
    city_reward -= (4 - game.state.player_state[f"{p1_key}_CITIES_AVAILABLE"])
    city_reward *= 100

    road_reward = game.state.player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    road_reward -= game.state.player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
    if road_reward < 2 and road_reward > -2:
        road_reward *= 2
    if game.state.player_state[f"{p_key}_HAS_ROAD"]:
        road_reward += 45
    elif game.state.player_state[f"{p1_key}_HAS_ROAD"]:
        road_reward -= 45

    largest_army_reward = game.state.player_state[f"{p_key}_PLAYED_KNIGHT"]
    largest_army_reward -= game.state.player_state[f"{p1_key}_PLAYED_KNIGHT"]
    if largest_army_reward < 2 and largest_army_reward > -2:
        largest_army_reward *= 2
    if game.state.player_state[f"{p_key}_HAS_ARMY"]:
        largest_army_reward += 45
    elif game.state.player_state[f"{p1_key}_HAS_ARMY"]:
        road_reward -= 45

    # development_card_reward = endgame_development_card_reward(game.state, p_key)
    # # print(f"endgame development_card_reward : {development_card_reward}")
    #
    # resource_reward = endgame_resource_reward(game.state, p_key, p0_color)
    # # print(f"endgame resource_reward : {resource_reward}")

    # total_reward += vp_reward
    total_reward += winning_reward
    total_reward += settlement_reward
    total_reward += city_reward
    total_reward += road_reward
    total_reward += largest_army_reward
    # total_reward += development_card_reward
    # total_reward += resource_reward
    # print(f"end game reward = {(total_reward-400)/10}")
    # return (total_reward-400)/10
    return total_reward

# michael's attempt to create a good reward function for PPO agent
def simple_reward2(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    if game.state.player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"] < 2:
        return initial_stage_reward2(game, p0_color)

    winning_reward = get_winning_reward(game, p0_color)
    if winning_reward != 0:
        return winning_reward

    total_reward = 0
    production_reward = 0
    production_reward += calculate_resource_production_value_for_player(game.state, p0_color)
    production_reward -= calculate_resource_production_value_for_player(game.state, p1_color)

    settlement_reward = 1.6 * (5 - game.state.player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"])
    settlement_reward -= 1.6 * (5 - game.state.player_state[f"{p1_key}_SETTLEMENTS_AVAILABLE"])

    city_reward = 3 * (4 - game.state.player_state[f"{p_key}_CITIES_AVAILABLE"])
    city_reward -= 3 * (4 - game.state.player_state[f"{p1_key}_CITIES_AVAILABLE"])

    road_reward = 0.3 * game.state.player_state[f"{p_key}_LONGEST_ROAD_LENGTH"]
    road_reward -= 0.3 * game.state.player_state[f"{p1_key}_LONGEST_ROAD_LENGTH"]
    if game.state.player_state[f"{p_key}_HAS_ROAD"]:
        road_reward += 3
    elif game.state.player_state[f"{p1_key}_HAS_ROAD"]:
        road_reward -= 3

    development_card_reward = calc_development_card_reward(game.state, p_key)

    resource_reward = calc_resource_reward(game.state, p_key, p0_color)
    resource_reward -= calc_resource_reward(game.state, p1_key, p1_color)

    total_reward += production_reward
    total_reward += settlement_reward
    total_reward += city_reward
    total_reward += road_reward
    total_reward += development_card_reward
    total_reward += resource_reward
    return total_reward

# michael's attempt to create a good reward function for PPO agent
def simple_reward(game, p0_color):
    p1_color = Color.RED
    if p0_color == p1_color:
        p1_color = Color.BLUE
    p_key = player_key(game.state, p0_color)
    p1_key = player_key(game.state, p1_color)

    if game.state.player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"] < 3:
        return initial_stage_reward(game, p0_color)

    # num_nodes = (5 - game.state.player_state[f"{p_key}_SETTLEMENTS_AVAILABLE"])
    # num_nodes += 2 * (4 - game.state.player_state[f"{p_key}_CITIES_AVAILABLE"])
    # if num_nodes > 6:
    #     return end_stage_reward(game, p0_color)

    winning_reward = get_winning_reward(game, p0_color)
    if winning_reward != 0:
        return winning_reward

    # return 0
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

    # largest_army_reward = 0.3 * game.state.player_state[f"{p_key}_PLAYED_KNIGHT"]
    # largest_army_reward -= 0.3 * game.state.player_state[f"{p1_key}_PLAYED_KNIGHT"]
    # if game.state.player_state[f"{p_key}_HAS_ARMY"]:
    #     largest_army_reward += 2
    # elif game.state.player_state[f"{p1_key}_HAS_ARMY"]:
    #     road_reward -= 2
    # largest_army_reward *= 0.5

    development_card_reward = calc_development_card_reward(game.state, p_key)
    # development_card_reward -= calc_development_card_reward(game.state, p1_key)
    # development_card_reward *= 0.2
    # # print(f"midgame development_card_reward : {development_card_reward}")

    resource_reward = calc_resource_reward(game.state, p_key, p0_color)
    resource_reward -= calc_resource_reward(game.state, p1_key, p1_color)
    # print(f"midgame resource_reward : {resource_reward}")

    # reachability_sample = reachability_features(game, p0_color, 2)
    # features = [f"P0_0_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
    # reachable_production_at_zero = sum([reachability_sample[f] for f in features])
    # features = [f"P0_1_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
    # reachable_production_at_one = sum([reachability_sample[f] for f in features])
    #
    # enemy_reachability_sample = reachability_features(game, p1_color, 2)
    # features = [f"P0_0_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
    # enemy_reachable_production_at_zero = sum([enemy_reachability_sample[f] for f in features])
    # features = [f"P0_1_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
    # enemy_reachable_production_at_one = sum([enemy_reachability_sample[f] for f in features])


    # reachability_reward = 2*reachable_production_at_zero
    # reachability_reward -= 2 * enemy_reachable_production_at_zero
    # reachability_reward += reachable_production_at_one
    # reachability_reward -= enemy_reachable_production_at_zero

    # print(f"reachable_production_at_zero : {reachable_production_at_zero}")
    # print(f"reachable_production_at_one : {reachable_production_at_one}")
    total_reward += production_reward
    total_reward += settlement_reward
    total_reward += city_reward
    total_reward += road_reward
    # total_reward += largest_army_reward
    total_reward += development_card_reward
    total_reward += resource_reward
    # total_reward += reachability_reward
    # print(f"total_reward : {total_reward}")
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

def endgame_development_card_reward(state, p_key):
    dev_card_reward = 0
    dev_card_reward += 1 * state.player_state[f"{p_key}_KNIGHT_IN_HAND"]
    dev_card_reward += 2 * state.player_state[f"{p_key}_YEAR_OF_PLENTY_IN_HAND"]
    dev_card_reward += 3 * state.player_state[f"{p_key}_ROAD_BUILDING_IN_HAND"]
    dev_card_reward += 4 * state.player_state[f"{p_key}_MONOPOLY_IN_HAND"]
    dev_card_reward += 15 * state.player_state[f"{p_key}_VICTORY_POINT_IN_HAND"]
    # dev_card_reward += 2 * state.player_state[f"{p_key}_PLAYED_KNIGHT"]
    # dev_card_reward += 4 *  state.player_state[f"{p_key}_PLAYED_MONOPOLY"]
    # dev_card_reward += 8 *  state.player_state[f"{p_key}_PLAYED_ROAD_BUILDING"]
    # dev_card_reward += 8 *  state.player_state[f"{p_key}_PLAYED_YEAR_OF_PLENTY"]
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
    if state.player_state[f"{p_key}_WOOD_IN_HAND"] > 0 :
        missing_cards -= 1
    if state.player_state[f"{p_key}_BRICK_IN_HAND"] > 0 :
        missing_cards -= 1
    if state.player_state[f"{p_key}_SHEEP_IN_HAND"] > 0 :
        missing_cards -= 1
    if state.player_state[f"{p_key}_WHEAT_IN_HAND"] > 0 :
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

