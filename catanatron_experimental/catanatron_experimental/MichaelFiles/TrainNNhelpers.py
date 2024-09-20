import numpy as np
import matplotlib.pyplot as plt


def load_numpy_data(filename):
    data = np.load(filename)
    X = data['X']
    Y = data['Y']
    return X, Y


def plot_losses(trn_los, val_los, plrs="FvF", gms=114000, tst=0.1, btch=16, lr=0.001, wght=1e-5):
    epochs = range(1, len(trn_los) + 1)
    plt.plot(epochs, trn_los, label='Train Loss')
    plt.plot(epochs, val_los, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    plt.title(f'{plrs}_{gms}, test={tst}, \nbatch={btch}, lr={lr}, weight_dec={wght}')

    plt.legend()
    plt.show()


def combine_datasets(players="MYVFvF", games=118000, num_features=485):

    # X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/WvW_1_99000games_364features.npz")
    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/WvW_2_99000games_364features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/WvW_22_99000games_364features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/WvW_turn_10_99000games_364features.npz")
    # X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/WvW_turn_20_99000games_364features.npz")

    X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/{players}_1_{games}games_{num_features}features.npz")
    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{players}_2_{games}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{players}_4_{games}games_{num_features}features.npz")
    X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{players}_6_{games}games_{num_features}features.npz")
    X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/{players}_8_{games}games_{num_features}features.npz")
    X_data5, Y_data5 = load_numpy_data(f"game_simulation_data/{players}_9_{games}games_{num_features}features.npz")

    X_data = np.concatenate((X_data0, X_data1, X_data2, X_data3, X_data4, X_data5), axis=0)
    Y_data = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5))

    # Y_data[Y_data == -1] = 0

    return X_data, Y_data


def get_1_datasets():
    num_features = 363

    fvf = "FvF"
    fvab = "FvAB"
    abvab = "ABvAB"
    wvw = "WvW"
    # X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/{fvf}_1_{114000}games_{num_features}features.npz")
    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvf}_1_{14000}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvf}_1_{4500}games_{num_features}features.npz")

    # X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvab}_1_{100}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvab}_1_{1300}games_{num_features}features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{fvab}_1_{1400}games_{num_features}features.npz")
    #
    # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))
    #
    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{abvab}_1_{350}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{abvab}_1_{700}games_{num_features}features.npz")

    # X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{wvw}_1_{5000}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{wvw}_1_{6000}games_{num_features}features.npz")
    X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{wvw}_1_{50000}games_{num_features}features.npz")

    # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))

    X_data0 = np.concatenate((X_data1, X_data2, X_data3), axis=0)
    Y_data0 = np.concatenate((Y_data1, Y_data2, Y_data3))
    Y_data0[Y_data0 == -1] = 0

    num_turns = X_data0[:, -1]
    print(f"mean turn for 1 dataset is: {np.mean(num_turns)}")
    print(f"min turn for 1 dataset is: {np.min(num_turns)}")
    print(f"max turn for 1 dataset is: {np.max(num_turns)}")

    return X_data0, Y_data0


def get_22_datasets():
    num_features = 363

    fvf = "FvF"
    fvab = "FvAB"
    abvab = "ABvAB"
    wvw = "WvW"
    X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/{fvf}_22_{114000}games_{num_features}features.npz")
    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvf}_22_{14000}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvf}_22_{4500}games_{num_features}features.npz")

    X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvab}_22_{100}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvab}_22_{1300}games_{num_features}features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{fvab}_22_{1400}games_{num_features}features.npz")
    #
    # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))
    #
    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{abvab}_22_{350}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{abvab}_22_{700}games_{num_features}features.npz")

    # X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{wvw}_22_{5000}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{wvw}_22_{6000}games_{num_features}features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{wvw}_22_{50000}games_{num_features}features.npz")

    # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))

    # X_data0 = np.concatenate((X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data1, Y_data2, Y_data3))

    Y_data0[Y_data0 == -1] = 0

    num_turns = X_data0[:, -1]
    print(f"mean turn for 22 dataset is: {np.mean(num_turns)}")
    print(f"min turn for 22 dataset is: {np.min(num_turns)}")
    print(f"max turn for 22 dataset is: {np.max(num_turns)}")

    return X_data0, Y_data0


def get_turn10_datasets():
    num_features = 363

    fvf = "FvF"
    fvab = "FvAB"
    abvab = "ABvAB"
    wvw = "WvW"
    X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/{fvf}_turn_10_{114000}games_{num_features}features.npz")
    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvf}_turn_10_{14000}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvf}_turn_10_{4500}games_{num_features}features.npz")

    X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvab}_turn_10_{100}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvab}_turn_10_{1300}games_{num_features}features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{fvab}_turn_10_{1400}games_{num_features}features.npz")
    #
    # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))
    #
    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{abvab}_turn_10_{350}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{abvab}_turn_10_{700}games_{num_features}features.npz")

    # X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{wvw}_turn_10_{5000}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{wvw}_turn_10_{6000}games_{num_features}features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{wvw}_turn_10_{50000}games_{num_features}features.npz")
    #
    # # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))
    #
    # X_data0 = np.concatenate((X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data1, Y_data2, Y_data3))

    Y_data0[Y_data0 == -1] = 0

    num_turns = X_data0[:, -1]
    print(f"mean turn for turn_10 dataset is: {np.mean(num_turns)}")
    print(f"min turn for turn_10 dataset is: {np.min(num_turns)}")
    print(f"max turn for turn_10 dataset is: {np.max(num_turns)}")

    return X_data0, Y_data0


def get_turn20_datasets():
    num_features = 363

    fvf = "FvF"
    fvab = "FvAB"
    abvab = "ABvAB"
    wvw = "WvW"
    X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/{fvf}_turn_20_{114000}games_{num_features}features.npz")
    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvf}_turn_20_{14000}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvf}_turn_20_{4500}games_{num_features}features.npz")

    X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{abvab}_turn_20_{350}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{abvab}_turn_20_{700}games_{num_features}features.npz")

    X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    # X_data0 = np.concatenate((X_data1, X_data2), axis=0)
    # Y_data0 = np.concatenate((Y_data1, Y_data2))

    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvab}_turn_20_{100}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvab}_turn_20_{1300}games_{num_features}features.npz")
    X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{fvab}_turn_20_{1400}games_{num_features}features.npz")

    X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))


    # X_data0 = np.concatenate((X_data1, X_data2), axis=0)
    # Y_data0 = np.concatenate((Y_data1, Y_data2))

    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{wvw}_turn_20_{5000}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{wvw}_turn_20_{6000}games_{num_features}features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{wvw}_turn_20_{50000}games_{num_features}features.npz")
    #
    # # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))
    #
    # X_data0 = np.concatenate((X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data1, Y_data2, Y_data3))

    Y_data0[Y_data0 == -1] = 0

    num_turns = X_data0[:, -1]
    print(f"mean turn for turn_20 dataset is: {np.mean(num_turns)}")
    print(f"min turn for turn_20 dataset is: {np.min(num_turns)}")
    print(f"max turn for turn_20 dataset is: {np.max(num_turns)}")
    return X_data0, Y_data0



def get_turn30_datasets():
    num_features = 363

    fvf = "FvF"
    fvab = "FvAB"
    abvab = "ABvAB"
    wvw = "WvW"
    # X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/{fvf}_turn_30_{114000}games_{num_features}features.npz")
    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvf}_turn_30_{14000}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvf}_turn_30_{4500}games_{num_features}features.npz")
    #
    # X_data0 = np.concatenate((X_data1, X_data2), axis=0)
    # Y_data0 = np.concatenate((Y_data1, Y_data2))

    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{fvab}_turn_30_{100}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{fvab}_turn_30_{1300}games_{num_features}features.npz")
    X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{fvab}_turn_30_{1400}games_{num_features}features.npz")

    X_data0 = np.concatenate((X_data1, X_data2, X_data3), axis=0)
    Y_data0 = np.concatenate((Y_data1, Y_data2, Y_data3))

    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{abvab}_turn_30_{350}games_{num_features}features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{abvab}_turn_30_{700}games_{num_features}features.npz")

    X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
    Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))

    # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{wvw}_turn_30_{5000}games_{num_features}features.npz")
    # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{wvw}_turn_30_{6000}games_{num_features}features.npz")
    # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{wvw}_turn_30_{50000}games_{num_features}features.npz")
    #
    # # X_data0 = np.concatenate((X_data0, X_data1, X_data2, X_data3), axis=0)
    # # Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3))
    #
    # X_data0 = np.concatenate((X_data1, X_data2, X_data3), axis=0)
    # Y_data0 = np.concatenate((Y_data1, Y_data2, Y_data3))

    Y_data0[Y_data0 == -1] = 0

    num_turns = X_data0[:, -1]
    print(f"mean turn for turn_30 dataset is: {np.mean(num_turns)}")
    print(f"min turn for turn_30 dataset is: {np.min(num_turns)}")
    print(f"max turn for turn_30 dataset is: {np.max(num_turns)}")
    return X_data0, Y_data0

def combine_two_datasets(X_data1, Y_data1, X_data2, Y_data2):

    X_data = np.concatenate((X_data1, X_data2), axis=0)
    Y_data = np.concatenate((Y_data1, Y_data2))

    return X_data, Y_data


def remove_VP_feature(X_data):
    X_data = np.delete(X_data, 324, axis=1)
    return np.delete(X_data, 343, axis=1)



