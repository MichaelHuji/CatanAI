import random
from catanatron.game import Game
from catanatron import Player
# from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental.MichaelFiles.HeuristicFeatures import simple_reward

from catanatron_gym.envs.catanatron_env import from_action_space

# @register_player("MYVF")
class MyVFPlayer(Player):
    def __init__(self, color, is_bot=True, epsilon=None):
        super().__init__(color, is_bot)
        self.epsilon = epsilon

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

            # for exploration
        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        best_value = float("-inf")
        best_actions = []
        for action in playable_actions:
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

        return random.choice(best_actions) # from all equally best actions randomly pick 1