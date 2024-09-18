from tqdm import tqdm
import numpy as np
from gymnasium import Env
from catanatron.models.player import Player
import gymnasium as gym
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.player import Color, Player, RandomPlayer

from catanatron_experimental.MichaelFiles.MyNNPlayer import MyNNPlayer
from catanatron_experimental.MichaelFiles.MyHeuristic import MyVFPlayer

def simulate_games(env, model, episodes, players):

    X1_data = []
    X2_data = []
    X22_data = []
    x_turn_10_data = []
    x_turn_20_data = []
    x_turn_30_data = []

    Y_data = []

    print(model)
    print(env.enemies)

    for episode in tqdm(range(episodes)):

        observation, info = env.reset()
        done = False

        X1 = np.zeros(324)
        X2 = np.zeros(324)
        X22 = np.zeros(324)
        x_turn_10 = np.zeros(324)
        x_turn_20 = np.zeros(324)
        x_turn_30 = np.zeros(324)
        Y = 0
        while not done:

            action = model.decide(env.game, env.get_valid_actions())
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            X1 = info['x1']
            X2 = info['x2']
            X22 = info['x22']
            x_turn_10 = info['turn_10']
            x_turn_20 = info['turn_20']
            x_turn_30 = info['turn_30']
            Y = info['y']

        X1_data.append(X1)
        X2_data.append(X2)
        X22_data.append(X22)
        x_turn_10_data.append(x_turn_10)
        x_turn_20_data.append(x_turn_20)
        x_turn_30_data.append(x_turn_30)

        Y_data.append(Y)

    np.savez(f"game_simulation_data/{players}_1_{episodes}games_363features.npz", X=np.array(X1_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_2_{episodes}games_363features.npz", X=np.array(X2_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_22_{episodes}games_363features.npz", X=np.array(X22_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_turn_10_{episodes}games_363features.npz", X=np.array(x_turn_10_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_turn_20_{episodes}games_363features.npz", X=np.array(x_turn_20_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_turn_30_{episodes}games_363features.npz", X=np.array(x_turn_30_data), Y=np.array(Y_data))

    return np.array(X22_data), np.array(Y_data)

env = gym.make("catanatron_gym:catanatron-v1")

# model = RandomPlayer(Color.BLUE)
# model = WeightedRandomPlayer(Color.BLUE)
# model = ValueFunctionPlayer(Color.BLUE)
# model = AlphaBetaPlayer(Color.BLUE)
model = MyNNPlayer(Color.BLUE)
# model = MyVFPlayer(Color.BLUE)


env.enemies = [RandomPlayer(Color.RED)]
env.enemies = [WeightedRandomPlayer(Color.RED)]
env.enemies = [ValueFunctionPlayer(Color.RED)]
env.enemies = [AlphaBetaPlayer(Color.RED)]
env.enemies = [MyNNPlayer(Color.RED)]
env.enemies = [MyVFPlayer(Color.RED)]

X_data, Y_data = simulate_games(env, model, 700, "NN0vNN0")  # generate data for {episodes} games

