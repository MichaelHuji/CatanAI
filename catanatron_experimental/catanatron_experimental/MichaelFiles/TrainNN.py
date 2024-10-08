import torch
import torch.nn as nn
from catanatron_experimental.MichaelFiles.TrainNNhelpers import combine_datasets, plot_losses
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm
from catanatron_experimental.MichaelFiles.Net import Net
from TrainNNhelpers import (combine_datasets, combine_two_datasets, get_22_datasets, get_turn10_datasets,
                            get_1_datasets, get_turn20_datasets, get_turn30_datasets, load_numpy_data)
import os


# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs,
                plrs="NN3vNN3", gms=114000, btch=16, lr=0.001):
    thrshold = 0.5
    train_losses = list()
    test_losses = list()
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model to training mode
        train_accuracy = 0.0
        train_size = 0
        for inputs, labels in train_loader:

            optimizer.zero_grad()  # Clear gradients from previous step
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights

            outputs[outputs >= thrshold] = 1
            outputs[outputs < thrshold] = 0
            train_size += len(labels)
            train_accuracy += torch.sum(torch.abs(outputs - labels)).item()
        print(f"Epoch [{epoch + 1}/{epochs}], train accuracy: {train_accuracy/train_size}, train_size: {train_size}")
        train_losses.append(train_accuracy/train_size)

        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_size = 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for inputs, labels in test_loader:
                outputs = model(inputs)

                outputs[outputs >= thrshold] = 1
                outputs[outputs < thrshold] = 0
                test_loss += torch.sum(torch.abs(outputs - labels)).item()
                test_size += len(labels)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {test_loss / test_size}, test size: {test_size}")
        test_losses.append(test_loss / test_size)

        torch.save(model.state_dict(), f'model_weights/{plrs}_{gms}_b{btch}_lr{lr}_363features_128d3_weights_epoch{epoch+1}.pth')

    return train_losses, test_losses

# Example: Splitting the data into train and test sets, training, and evaluating
def train_and_evaluate(model_class, X, Y, test_size=0.1, epochs=30,
                       plrs="NN3vNN3", gms=114000, batch_size=16, lr=0.001, weight_decay=1e-5,
                       weights=None):

    # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    X_train, Y_train = X, Y
    filename = f"game_simulation_data/"

    X_test, Y_test = load_numpy_data(filename + "FvF_22_14000games_363features.npz")
    # X_test1, Y_test1 = load_numpy_data(filename + "FvF_22_4500games_363features.npz")
    Y_test[Y_test == -1] = 0
    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoaders for training and testing
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = Net(input_size=363, layer1 = 128, layer2 = 64)

    if weights != None:
        # load pre-trained weights

        file_path = 'model_weights/' + weights

        model.load_state_dict(torch.load(file_path))
        print('loaded weights')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    trn_los, tst_los = train_model(model, train_loader, test_loader, criterion, optimizer, epochs,
                plrs, gms, batch_size, lr)

    return trn_los, tst_los



# X_data, Y_data = get_22_datasets()
# X_data, Y_data = get_turn10_datasets()
# X_data, Y_data = get_turn20_datasets()
# X_data, Y_data = get_turn30_datasets()

# f"game_simulation_data/{players}_1_{games}games_{num_features}features.npz"
filename = f"game_simulation_data/"
#
# X_data0, Y_data0 = load_numpy_data(filename + "WvW_22_5000games_363features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + "WvW_22_6000games_363features.npz")
# # X_data0 = np.concatenate((X_data0, X_data1), axis=0)
# # Y_data0 = np.concatenate((Y_data0, Y_data1))
#
# X_data2, Y_data2 = load_numpy_data(filename + "WvW_22_50000games_363features.npz")
# X_data0 = np.concatenate((X_data0, X_data1, X_data2), axis=0)
# Y_data0 = np.concatenate((Y_data0, Y_data1, Y_data2))
#
# X_data1, Y_data1 = load_numpy_data(filename + "FvF_22_114000games_363features.npz")
# #
# # # X_data0 = X_data0[60000:, :]
# # # Y_data0 = Y_data0[60000:, :]
# X_data2, Y_data2 = load_numpy_data(filename + "FvF_22_14000games_363features.npz")
#
# X_data = np.concatenate((X_data0, X_data1, X_data2), axis=0)
# Y_data = np.concatenate((Y_data0, Y_data1, Y_data2))

# X_data = np.concatenate((X_data0, X_data1, X_data2), axis=0)
# Y_data = np.concatenate((Y_data0, Y_data1, Y_data2))

# #
# X_data, Y_data = load_numpy_data(filename + "FvF_22_150000games_364features.npz")
# X_data = X_data[:10000, :]
# Y_data = Y_data[:10000]
# X_data1, Y_data1 = load_numpy_data(filename + "F.5vF.5_2_12000games_363features.npz")
#
# X_data = np.concatenate((X_data, X_data1), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1))

# X_data = np.concatenate((X_data, X_data1, X_data2), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1, Y_data2))
# X_data = X_data[:60000, :]
# Y_data = Y_data[:60000]

# X_data, Y_data = load_numpy_data(filename + "MYVF.2vMYVF.2_2_40000games_363features.npz")


X_data, Y_data = load_numpy_data(filename + "MYVF.5vMYVF.5_2_78000games_363features.npz")

# X_data = np.concatenate((X_data, X_data1), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1))

games = len(Y_data)


Y_data[Y_data == -1] = 0


players = "MYVF.5vMYVF.5_2"   # choose on which player data to train the model
test_size = 0.1             # choose what percent of data is used in the test set
epochs = 30                 # choose the number of epochs to train the model
batch_size = 32              # choose the size of batches used in training
learning_rate = 0.001        # choose the learning rate
weight_decay = 1e-5         # choose the weight decay
weights = None              # choose the weights to load in the model before the training
# weights = "MYVF.2vMYVF.2_2__40000_b32_lr0.001_363features_128d3_weights_epoch30.pth"

# # Train and evaluate the model using 80% for training and 20% for testing
train_losses, test_losses = train_and_evaluate(Net, X_data,
                                               Y_data,
                                               test_size=test_size,
                                               epochs=epochs,
                                               plrs=players,
                                               gms=games,
                                               batch_size=batch_size,
                                               lr=learning_rate,
                                               weight_decay = weight_decay,
                                               weights=weights
                                               )

plot_losses(train_losses, test_losses, plrs=players,
            gms=games, tst=test_size,
            btch=batch_size, lr=learning_rate, wght=weight_decay)
