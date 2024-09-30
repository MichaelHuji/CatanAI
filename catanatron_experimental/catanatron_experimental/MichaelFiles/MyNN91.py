import random
import torch
import torch.nn as nn
from catanatron.game import Game
from catanatron import Player
# from catanatron_experimental.cli.cli_players import register_player
from catanatron_experimental.MichaelFiles.Net import Net
from catanatron_experimental.MichaelFiles.Net91 import Net91
from catanatron_experimental.MichaelFiles.Features import generate_x
from catanatron_experimental.MichaelFiles.VF_weights_features import generate_feature_vector
from catanatron.state_functions import player_key
from catanatron_experimental.MichaelFiles.ActionsSpace import from_action_space
from catanatron.models.player import Color
import os

class MyNN91(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    def __init__(self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None):

        super().__init__(color, is_bot)

        weight_file_path = 'catanatron_experimental/catanatron_experimental/MichaelFiles/model_weights/'

        weight_file_name1 = "F5_4_50000_b32_lr0.01_91features_weights_epoch30.pth"  # 0.491 64% 8.89 7.48 ****
        
        weight_file_name2 = "F5_4_50000_b32_lr0.01_91features_weights_epoch50.pth"  # 0.490 61% 8.67 7.27 ****
        
        weight_file_name3 = "all_data_4_465000_b64_lr0.004_91features_weights_epoch75.pth"  # 58% 8.75 7.71  ***
        
        weight_file_name4 = "all_F_2and4_630000_b64_lr0.01_91features_weights_epoch60.pth"  # 59% 8.40 7.62 ***
        
        weight_file_name5 = "F1vF1_4_150000_b32_lr0.01_91features_weights_epoch69.pth"  # 61% 8.54 7.56    ****
        
        weight_file_name6 = "F5_6_with_pretrained_F5_4_50000_b64_lr0.0001_91features_weights_epoch6.pth"  # as late: 63% 8.64 7.31  ****
        
        weight_file_name7 = "F_4_with_pretrained_F5_4_105000_b128_lr1e-06_91features_weights_epoch1.pth"  # 61% 8.68 7.57  ****

        self.model1 = Net91()
        self.model1.load_state_dict(torch.load(weight_file_path + weight_file_name1))
        self.model1.eval()

        self.model2 = Net91()
        self.model2.load_state_dict(torch.load(weight_file_path + weight_file_name2))
        self.model2.eval()

        self.model3 = Net91()
        self.model3.load_state_dict(torch.load(weight_file_path + weight_file_name3))
        self.model3.eval()

        self.model4 = Net91()
        self.model4.load_state_dict(torch.load(weight_file_path + weight_file_name4))
        self.model4.eval()

        self.model5 = Net91()
        self.model5.load_state_dict(torch.load(weight_file_path + weight_file_name5))
        self.model5.eval()

        self.model6 = Net91()
        self.model6.load_state_dict(torch.load(weight_file_path + weight_file_name6))
        self.model6.eval()

        self.model7 = Net91()
        self.model7.load_state_dict(torch.load(weight_file_path + weight_file_name7))
        self.model7.eval()


    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        p0_color = self.color
        p1_color = Color.RED
        if p0_color == p1_color:
            p1_color = Color.BLUE
        p_key = player_key(game.state, p0_color)
        p1_key = player_key(game.state, p1_color)

        state = game.state
        board = state.board
        player_state = state.player_state

        best_value = float("-inf")
        best_actions = []

        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            action_vector = generate_feature_vector(game_copy, self.color)
            action_value = 0
            # action_value += self.model1(torch.tensor(action_vector, dtype=torch.float32))
            
            action_value += self.model2(torch.tensor(action_vector, dtype=torch.float32))
            
            # action_value += self.model3(torch.tensor(action_vector, dtype=torch.float32))
            # action_value += self.model4(torch.tensor(action_vector, dtype=torch.float32))
            # action_value += self.model5(torch.tensor(action_vector, dtype=torch.float32))
            # action_value += self.model6(torch.tensor(action_vector, dtype=torch.float32))
            # action_value += self.model7(torch.tensor(action_vector, dtype=torch.float32))

            # if ((player_state[f"{p_key}_ACTUAL_VICTORY_POINTS"] >= 7) or
            #                 (player_state[f"{p1_key}_ACTUAL_VICTORY_POINTS"] >= 7)):
            #     action_value += self.model6(torch.tensor(action_vector, dtype=torch.float32))

            if action_value == best_value:
                best_actions.append(action)
            if action_value > best_value:
                best_value = action_value
                best_actions = [action]

        return random.choice(best_actions)
