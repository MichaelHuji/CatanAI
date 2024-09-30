import random
import torch
import torch.nn as nn
from catanatron.game import Game
from catanatron import Player
# from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental.MichaelFiles.Net import Net
# from catanatron_experimental.MichaelFiles.Net91 import Net91
from catanatron_experimental.MichaelFiles.Features import generate_x
from catanatron_experimental.MichaelFiles.ActionsSpace import from_action_space
import os


class MyNNPlayer(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):

        super().__init__(color, is_bot)


        weight_file_path = 'catanatron_experimental/catanatron_experimental/MichaelFiles/model_weights/'

        weight_file_name8 = "WvW_60000_b32_lr0.001_363features_128d_weights_epoch13.pth"      # NN8 - good 4.79 vp vs F
        self.model8 = Net(input_size=363, layer1=128, layer2=64)
        self.model8.load_state_dict(torch.load(weight_file_path + weight_file_name8))

        weight_file_name24 = "FvF_10_128000_b32_lr0.001_363features_128d3_weights_epoch21.pth"      # NN 24 - good 5.187 vp vs F
        self.model24 = Net(input_size=363, layer1=128, layer2=64)
        self.model24.load_state_dict(torch.load(weight_file_path + weight_file_name24))

        weight_file_name25 = "WvW_10_61000_b32_lr0.001_363features_128d3_weights_epoch10.pth"      # NN 25 - good 4.97 vp vs F
        self.model25 = Net(input_size=363, layer1=128, layer2=64)
        self.model25.load_state_dict(torch.load(weight_file_path + weight_file_name25))

        weight_file_name26 = "FvF_plus_WvW_10_189000_b32_lr0.001_363features_128d3_weights_epoch16.pth" # NN 26 - good 5.63 vp vs F
        self.model26 = Net(input_size=363, layer1 = 128, layer2 = 64)
        self.model26.load_state_dict(torch.load(weight_file_path + weight_file_name26))

        weight_file_name27 = "FvF_plus_WvW_20_189000_b32_lr0.001_363features_128d3_weights_epoch17.pth" # NN 27
        self.model27 = Net(input_size=363, layer1 = 128, layer2 = 64)
        self.model27.load_state_dict(torch.load(weight_file_path + weight_file_name27))


    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_actions = []

        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            action_vector = generate_x(game_copy, self.color)
            action_value8 = self.model8(torch.tensor(action_vector, dtype=torch.float32))
            action_value24 = self.model24(torch.tensor(action_vector, dtype=torch.float32))
            action_value25 = self.model25(torch.tensor(action_vector, dtype=torch.float32))
            action_value26 = self.model26(torch.tensor(action_vector, dtype=torch.float32))
            action_value27 = self.model27(torch.tensor(action_vector, dtype=torch.float32))
            action_value = action_value8 + action_value24 + action_value25 + action_value26 + action_value27
            if action_value == best_value:
                best_actions.append(action)
            if action_value > best_value:
                best_value = action_value
                best_actions = [action]

        return random.choice(best_actions)