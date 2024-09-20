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


# @register_player("NN")
class MyNNPlayer(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):

        super().__init__(color, is_bot)

        # # Get current working directory
        # cwd = os.getcwd()

        weight_file_name = 'catanatron_experimental/catanatron_experimental/MichaelFiles/model_weights/'

        # weight_file_name0 = "best_model_weights_attempt2.pth"     # Best weights 20.9.2024 , 19:25
        # self.model = Net()
        # self.model.load_state_dict(torch.load(weight_file_name + weight_file_name0))


        weight_file_name1 = "WvW_22_61000_b32_lr0.01_363features_weights_epoch5.pth"
        self.model_early = Net()
        self.model_early.load_state_dict(torch.load(weight_file_name + weight_file_name1))

        weight_file_name2 = "WvW_turn10_61000_b32_lr0.01_363features_weights_epoch7.pth"
        self.model_mid = Net()
        self.model_mid.load_state_dict(torch.load(weight_file_name + weight_file_name2))

        # weight_file_name3 = "WvW_turn20_61000_b32_lr0.01_363features_weights_epoch6.pth"
        # weight_file_name3 = "FvF_turn20_132500_b16_lr0.005_363features_weights_epoch7.pth"
        weight_file_name3 = "FvF_turn20_136350_b64_lr1e-05_363features_weights_epoch26.pth"
        self.model_mid_late = Net()
        self.model_mid_late.load_state_dict(torch.load(weight_file_name + weight_file_name3))

        weight_file_name4 = "WvW_turn30_61000_b32_lr0.01_363features_weights_epoch10.pth"
        self.model_late = Net()
        self.model_late.load_state_dict(torch.load(weight_file_name + weight_file_name4))



    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_actions = []

        # current_model = self.model
        current_model = self.model_early
        if game.state.num_turns > 3 and game.state.num_turns < 13:
            current_model = self.model_mid
        elif game.state.num_turns > 12 and game.state.num_turns < 22:
            current_model = self.model_mid_late
        elif game.state.num_turns >= 22:
            current_model = self.model_late

        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            action_vector = generate_x(game_copy, self.color)

            action_value = current_model(torch.tensor(action_vector, dtype=torch.float32))
            if action_value == best_value:
                best_actions.append(action)
            if action_value > best_value:
                best_value = action_value
                best_actions = [action]

        return random.choice(best_actions)
