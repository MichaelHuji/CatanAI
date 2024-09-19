from tqdm import tqdm
import numpy as np
from gymnasium import Env
from catanatron.models.player import Player
import gymnasium as gym
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron_gym.envs.catanatron_env import CatanatronEnv

from catanatron_experimental.MichaelFiles.MyNNPlayer import MyNNPlayer
from catanatron_experimental.MichaelFiles.MyHeuristic import MyVFPlayer

def simulate_games(env, model, episodes, players):

    X1_data = []
    X2_data = []
    X4_data = []
    X6_data = []
    X8_data = []
    X9_data = []

    Y_data = []

    print(model)
    print(env.enemies)

    for episode in tqdm(range(episodes)):

        observation, info = env.reset()
        done = False

        X1 = np.zeros(477)
        X2 = np.zeros(477)
        X4 = np.zeros(477)
        X6 = np.zeros(477)
        X8 = np.zeros(477)
        X9 = np.zeros(477)
        Y = 0
        while not done:

            action = model.decide(env.game, env.get_valid_actions())
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            X1 = info['x1']
            X2 = info['x2']
            X4 = info['x4']
            X6 = info['x6']
            X8 = info['x8']
            X9 = info['x9']
            Y = info['y']

        # normalize the turn feature
        num_turns = env.game.state.num_turns + 1
        X1[476] /= num_turns
        X2[476] /= num_turns
        X4[476] /= num_turns
        X6[476] /= num_turns
        X8[476] /= num_turns
        X9[476] /= num_turns
        X1_data.append(X1)
        X2_data.append(X2)
        X4_data.append(X4)
        X6_data.append(X6)
        X8_data.append(X8)
        X9_data.append(X9)

        Y_data.append(Y)

    # print(X1_data)
    # print(Y_data)
    # print(np.array(X9_data)[300:500, 476])
    np.savez(f"game_simulation_data/{players}_1_{episodes}games_477features.npz", X=np.array(X1_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_2_{episodes}games_477features.npz", X=np.array(X2_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_4_{episodes}games_477features.npz", X=np.array(X4_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_6_{episodes}games_477features.npz", X=np.array(X6_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_8_{episodes}games_477features.npz", X=np.array(X8_data), Y=np.array(Y_data))

    np.savez(f"game_simulation_data/{players}_9_{episodes}games_477features.npz", X=np.array(X9_data), Y=np.array(Y_data))

    return np.array(X9_data), np.array(Y_data)

env = gym.make("catanatron_gym:catanatron-v1")
# env = CatanatronEnv()
# model = RandomPlayer(Color.BLUE)
# model = WeightedRandomPlayer(Color.BLUE)
# model = ValueFunctionPlayer(Color.BLUE, epsilon=0.15)
# model = AlphaBetaPlayer(Color.BLUE)
# model = MyNNPlayer(Color.BLUE)
model = MyVFPlayer(Color.BLUE, epsilon=0.1)


# env.enemies = [RandomPlayer(Color.RED)]
# env.enemies = [WeightedRandomPlayer(Color.RED)]
# env.enemies = [ValueFunctionPlayer(Color.RED)]
# env.enemies = [AlphaBetaPlayer(Color.RED)]
# env.enemies = [MyNNPlayer(Color.RED)]
# env.enemies = [MyVFPlayer(Color.RED)]
# env.players = [env.p0] + env.enemies

num_games = 115000
X_data, Y_data = simulate_games(env, model, num_games, "MYVF.1evF.1e")  # generate data for {episodes} games

print(f"np.sum(Y_data) : {np.sum(Y_data)}")
print(f"np.count_nonzero(Y_data) : {np.count_nonzero(Y_data)}")
print(f"BLUE (MYVF 0.1 epsilon) wins: {num_games-np.count_nonzero(1-Y_data)}")
print(f"RED (F 0.1 epsilon)  wins: {num_games-np.count_nonzero(Y_data)}")
print(X_data.shape)
# print(X_data)