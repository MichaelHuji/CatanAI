import numpy as np


from TrainNNhelpers import load_numpy_data

def add_8_features_for_each_player(X):
    # Calculate the new features
    number_of_buildings = X[:, 2] + X[:, 3]
    wood_brick_production_relation = np.abs(X[:, 15] - X[:, 16])
    wheat_ore_production_relation = np.abs((X[:, 18] / 2) - (X[:, 19] / 3))
    wood_port_times_production = X[:, 15] * X[:, 24]
    brick_port_times_production = X[:, 16] * X[:, 25]
    sheep_port_times_production = X[:, 17] * X[:, 26]
    wheat_port_times_production = X[:, 18] * X[:, 27]
    ore_port_times_production = X[:, 19] * X[:, 28]

    # Stack the new columns to the original array
    new_entries = np.column_stack((number_of_buildings, wood_brick_production_relation,
                                   wheat_ore_production_relation, wood_port_times_production,
                                   brick_port_times_production, sheep_port_times_production,
                                   wheat_port_times_production, ore_port_times_production,
                                   ))
    X = np.hstack((X, new_entries))

    p1_number_of_buildings = X[:, 39] + X[:, 40]
    p1_wood_brick_production_relation = np.abs(X[:, 52] - X[:, 53])
    p1_wheat_ore_production_relation = np.abs((X[:, 55] / 2) - (X[:, 56] / 3))
    p1_wood_port_times_production = X[:, 52] * X[:, 61]
    p1_brick_port_times_production = X[:, 53] * X[:, 62]
    p1_sheep_port_times_production = X[:, 54] * X[:, 63]
    p1_wheat_port_times_production = X[:, 55] * X[:, 64]
    p1_ore_port_times_production = X[:, 56] * X[:, 65]

    # Stack the new columns to the original array
    p1_new_entries = np.column_stack((p1_number_of_buildings, p1_wood_brick_production_relation,
                                   p1_wheat_ore_production_relation, p1_wood_port_times_production,
                                   p1_brick_port_times_production, p1_sheep_port_times_production,
                                   p1_wheat_port_times_production, p1_ore_port_times_production,
                                   ))
    X = np.hstack((X, p1_new_entries))
    X[31] = 0
    return X


filename = f"game_simulation_data/"

#
# # ---------------------------------------------- W ----------------------------------------------
# print("\nCREATING COMBINED DATASET FOR W 2")
# players = "WvW"
# stage = "2"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_10000games_75features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_5000games_75features.npz")
# X_data2, Y_data2 = load_numpy_data(filename + f"{players}_{stage}_100000games_75features.npz")
# X_data3, Y_data3 = load_numpy_data(filename + f"{players}_{stage}_35000games_75features.npz")
# X_data = np.concatenate((X_data, X_data1, X_data2, X_data3), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3))
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
# print("\nCREATING COMBINED DATASET FOR W 4")
# players = "WvW"
# stage = "4"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_10000games_75features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_5000games_75features.npz")
# X_data2, Y_data2 = load_numpy_data(filename + f"{players}_{stage}_100000games_75features.npz")
# X_data3, Y_data3 = load_numpy_data(filename + f"{players}_{stage}_35000games_75features.npz")
# X_data = np.concatenate((X_data, X_data1, X_data2, X_data3), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3))
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
#
# # # ---------------------------------------------- F ----------------------------------------------
# print("\nCREATING COMBINED DATASET FOR F 2")
# players = "FvF"
# stage = "2"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_105000games_75features.npz")
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
#
# print("\nCREATING COMBINED DATASET FOR F 4")
# players = "FvF"
# stage = "4"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_105000games_75features.npz")
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
# # # ---------------------------------------------- F3 ----------------------------------------------
# print("\nCREATING COMBINED DATASET FOR F3 2")
# players = "F3"
# stage = "2"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_110000games_75features.npz")
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
# print("\nCREATING COMBINED DATASET FOR F3 4")
# players = "F3"
# stage = "4"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_110000games_75features.npz")
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
#
# # ---------------------------------------------- F2 ----------------------------------------------
# print("\nCREATING COMBINED DATASET FOR F2 2")
# players = "F2"
# stage = "2"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_17000games_75features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_33000games_75features.npz")
# X_data = np.concatenate((X_data, X_data1), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1))
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
# print("\nCREATING COMBINED DATASET FOR F2 4")
# players = "F2"
# stage = "4"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_17000games_75features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_33000games_75features.npz")
# X_data = np.concatenate((X_data, X_data1), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1))
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
# #
#
# # # ---------------------------------------------- F5 ----------------------------------------------
# print("\nCREATING COMBINED DATASET FOR F5 2")
# players = "F5vF5"
# stage = "2"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_15000games_75features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_35000games_75features.npz")
# X_data = np.concatenate((X_data, X_data1), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1))
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))
#
# print("\nCREATING COMBINED DATASET FOR F5 4")
# players = "F5vF5"
# stage = "4"
# X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_15000games_75features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_35000games_75features.npz")
# X_data = np.concatenate((X_data, X_data1), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1))
#
# X_data = add_8_features_for_each_player(X_data)
# print(X_data.shape)
# print(Y_data.shape)
# games = len(Y_data)
#
# np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))



print("\nCREATING COMBINED DATASET FOR F5 3")
players = "F5vF5"
stage = "3"
X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_15000games_75features.npz")
X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_35000games_75features.npz")
X_data = np.concatenate((X_data, X_data1), axis=0)
Y_data = np.concatenate((Y_data, Y_data1))

X_data = add_8_features_for_each_player(X_data)
print(X_data.shape)
print(Y_data.shape)
games = len(Y_data)

np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))


print("\nCREATING COMBINED DATASET FOR F5 5")
players = "F5vF5"
stage = "5"
X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_15000games_75features.npz")
X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_35000games_75features.npz")
X_data = np.concatenate((X_data, X_data1), axis=0)
Y_data = np.concatenate((Y_data, Y_data1))

X_data = add_8_features_for_each_player(X_data)
print(X_data.shape)
print(Y_data.shape)
games = len(Y_data)

np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))

print("\nCREATING COMBINED DATASET FOR F5 6")
players = "F5vF5"
stage = "6"
X_data, Y_data = load_numpy_data(filename + f"{players}_{stage}_15000games_75features.npz")
X_data1, Y_data1 = load_numpy_data(filename + f"{players}_{stage}_35000games_75features.npz")
X_data = np.concatenate((X_data, X_data1), axis=0)
Y_data = np.concatenate((Y_data, Y_data1))

X_data = add_8_features_for_each_player(X_data)
print(X_data.shape)
print(Y_data.shape)
games = len(Y_data)

np.savez(f"game_simulation_data/combined_{players}_{stage}_{games}games_91features.npz", X=np.array(X_data), Y=np.array(Y_data))





