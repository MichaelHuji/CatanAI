import random
import torch
import torch.nn as nn
from catanatron.game import Game
from catanatron import Player
# from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental.MichaelFiles.Net import Net
from catanatron_experimental.MichaelFiles.Features import generate_x
from catanatron_experimental.MichaelFiles.ActionsSpace import from_action_space
import os

# best_weights = f'NN2vNN2_47K_b16_lr005_model_weights_epoch19.pth'
# DEFAULT_WEIGHT='C:/Users/micha/PycharmProjects/catanProj/catanProj/catanatron_experimental/catanatron_experimental/MichaelFiles/model_weights/'
# # DEFAULT_WEIGHT='./model_weights/'

some_weight_file = 'MYVF.2evF.25e_11004_b16_lr0.001_weights_epoch7.pth'
some_weight_file2 = 'MYVF.2evF.25e_11004_b16_lr0.001_weights_epoch10.pth'
# @register_player("NN")
class MyNNPlayer(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):

        super().__init__(color, is_bot)

        # Get current working directory
        cwd = os.getcwd()

        # weight_file_name = f'NN2vNN2_47K_b16_lr005_model_weights_epoch19.pth' # is the best model we found so far
        weight_file_name = 'catanatron_experimental/catanatron_experimental/MichaelFiles/model_weights/'
        # weight_file_name += "MYVF.1evF.1e_115000_b8_lr0.001_weights_epoch12.pth"
        # weight_file_name += "MYVF.1evF.1e_115000_b16_lr0.01_weights_epoch7.pth"
        weight_file_name += "MYVFvF_60000_b32_lr0.001_weights_epoch5.pth"
        # Join the directory with the file name
        file_path = os.path.join(cwd, weight_file_name)

        self.weights = file_path
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