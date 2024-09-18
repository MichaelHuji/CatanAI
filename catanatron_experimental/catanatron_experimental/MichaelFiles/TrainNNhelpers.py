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


def combine_datasets(players="FvF", games=114000):
    X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/{players}_1_{games}games_363features.npz")
    X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/{players}_2_{games}games_363features.npz")
    X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/{players}_22_{games}games_363features.npz")
    X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/{players}_turn_10_{games}games_363features.npz")
    X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/{players}_turn_20_{games}games_363features.npz")

    X_data = np.concatenate((X_data0, X_data1, X_data2, X_data3, X_data4), axis=0)
    Y_data = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3, Y_data4))

    Y_data[Y_data == -1] = 0

    return X_data, Y_data