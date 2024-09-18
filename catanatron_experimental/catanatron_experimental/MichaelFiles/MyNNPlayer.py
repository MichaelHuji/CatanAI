import random
import torch
import torch.nn as nn
from catanatron.game import Game
from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental.MichaelFiles.Net import Net
from catanatron_experimental.MichaelFiles.Features import generate_x
from catanatron_experimental.MichaelFiles.ActionsSpace import from_action_space

# best_weights = f'NN2vNN2_47K_b16_lr005_model_weights_epoch19.pth'
DEFAULT_WEIGHT='FvF_all_129K_363feat_model_weights_epoch39.pth'

# @register_player("NNN")
class MyNNPlayer(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):

        super().__init__(color, is_bot)

        self.weights = DEFAULT_WEIGHT
        self.model = Net()
        # f'NN2vNN2_47K_b16_lr005_model_weights_epoch19.pth' is the best model we found so far
        self.model.load_state_dict(torch.load(self.weights))
        self.model.eval()

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_actions = []
        for action in playable_actions:
            game_copy = game.copy()
            if isinstance(action, int):
                catan_action = from_action_space(action, game.state.playable_actions)
                game_copy.execute(catan_action)

            else:
                game_copy.execute(action)

            action_vector = generate_x(game_copy, self.color)

            action_value = self.model(torch.tensor(action_vector, dtype=torch.float32))
            if action_value == best_value:
                best_actions.append(action)
            if action_value > best_value:
                best_value = action_value
                best_actions = [action]

        return random.choice(best_actions)

