import random

from catanatron.models.player import Player
from catanatron.models.actions import ActionType
from catanatron.state_functions import player_key

WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}

# WEIGHTS_BY_ACTION_TYPE = {
#     ActionType.BUILD_CITY: 1000,
#     ActionType.BUILD_SETTLEMENT: 1000,
#     ActionType.BUY_DEVELOPMENT_CARD: 0,
#     ActionType.BUILD_ROAD: 5
# }


from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES, LandTile, BASE_MAP_TEMPLATE

from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, Action, ActionType

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


class WeightedRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution
    to actions that are likely better (cities > settlements > dev cards).
    """

    def decide(self, game, playable_actions):

        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)

    # def decide(self, game, playable_actions):
    #
    #     if len(playable_actions) == 1:
    #         return playable_actions[0]
    #
    #     best_value = float("-inf")
    #     best_actions = []
    #     for action in playable_actions:
    #         game_copy = game.copy()
    #         if isinstance(action, int):
    #             catan_action = from_action_space(action, game.state.playable_actions)
    #             game_copy.execute(catan_action)
    #
    #         else:
    #             game_copy.execute(action)
    #
    #         key = player_key(game_copy.state, self.color)
    #         value = game_copy.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    #         if value == best_value:
    #             best_actions.append(action)
    #         if value > best_value:
    #             best_value = value
    #             best_actions = [action]

        # bloated_actions = []
        # for action in best_actions:
        #     if isinstance(action, int):
        #         catan_action = from_action_space(action, game.state.playable_actions)
        #         weight = WEIGHTS_BY_ACTION_TYPE.get(catan_action.action_type, 1)
        #     else:
        #         weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
        #     bloated_actions.extend([action] * weight)
        #
        # return random.choice(bloated_actions)
