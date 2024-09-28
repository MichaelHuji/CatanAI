import torch
import torch.nn as nn
from catanatron_experimental.MichaelFiles.TrainNNhelpers import combine_datasets, plot_losses
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm
from catanatron_experimental.MichaelFiles.Net91 import Net91
from TrainNNhelpers import (combine_datasets, combine_two_datasets, get_22_datasets, get_turn10_datasets,
                            get_1_datasets, get_turn20_datasets, get_turn30_datasets, load_numpy_data)
import os


# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs,
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

        torch.save(model.state_dict(), f'model_weights/{plrs}_{gms}_b{btch}_lr{lr}_91features_weights_epoch{epoch+1}.pth')
        scheduler.step()

    return train_losses, test_losses

# Example: Splitting the data into train and test sets, training, and evaluating
def train_and_evaluate(model_class, X, Y, test_size=0.1, epochs=30,
                       plrs="NN3vNN3", gms=114000, batch_size=16, lr=0.001, weight_decay=1e-5,
                       weights=None):

    # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    X_train, Y_train = X, Y
    filename = f"game_simulation_data/"
    X_test, Y_test = load_numpy_data(filename + "combined_FvF_4_105000games_91features.npz")
    # X_test = X_test[50000:, :]
    # Y_test = Y_test[50000:]
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
    model = Net91()
    # Manually set the weights
    # with torch.no_grad():
    #     model.output.weight = nn.Parameter(torch.ones_like(model.output.weight))  # Set all weights to 1
    #     model.output.bias = nn.Parameter(torch.zeros_like(model.output.bias))  # Set all biases to 0

    if weights != None:
        # load pre-trained weights

        file_path = 'model_weights/' + weights

        model.load_state_dict(torch.load(file_path))
        print('loaded weights')

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define scheduler (reduce lr by a factor of 0.1 every 10 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # Define loss function
    criterion = nn.BCELoss()

    trn_los, tst_los = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs,
                plrs, gms, batch_size, lr)

    model_output_weights = model.output.weight
    for i in range(91):
        print(f"feature {i} =\t{100*model_output_weights[0, i]:.3f}")

    # Print weights and biases of the 'fc' layer
    # print("Weights of fc layer:\n", model.output.weight)
    # print("Biases of fc layer:\n", model.output.bias)

    return trn_los, tst_los


filename = f"game_simulation_data/"
#
# X_data, Y_data = load_numpy_data(filename + "combined_F5vF5_2_50000games_91features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + "combined_F2_2_50000games_91features.npz")
# X_data2, Y_data2 = load_numpy_data(filename + "combined_F3_2_110000games_91features.npz")
# X_data3, Y_data3 = load_numpy_data(filename + "combined_FvF_2_105000games_91features.npz")
# X_data = np.concatenate((X_data, X_data1, X_data2, X_data3), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3))


# X_data4, Y_data4 = load_numpy_data(filename + "combined_F5vF5_4_50000games_91features.npz")
# X_data1, Y_data1 = load_numpy_data(filename + "combined_F2_4_50000games_91features.npz")
# X_data2, Y_data2 = load_numpy_data(filename + "combined_F3_4_110000games_91features.npz")
# X_data3, Y_data3 = load_numpy_data(filename + "combined_FvF_4_105000games_91features.npz")
# X_data, Y_data = load_numpy_data(filename + "combined_WvW_4_150000games_91features.npz")
# X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4))


X_data, Y_data = load_numpy_data(filename + "combined_F5vF5_4_50000games_91features.npz")
# X_data = X_data[:50000, :]
# Y_data = Y_data[:50000]
# X_data, Y_data = load_numpy_data(filename + "F1vF1_5_150000games_75features.npz")
games = len(Y_data)

Y_data[Y_data == -1] = 0
print(X_data.shape)
players = "F5_4_with_pretrained_all_F_4"   # choose on which player data to train the model
test_size = 0.1             # choose what percent of data is used in the test set
epochs = 50                 # choose the number of epochs to train the model
batch_size = 8              # choose the size of batches used in training
learning_rate = 0.000001        # choose the learning rate
weight_decay = 1e-4         # choose the weight decay
# weights = None              # choose the weights to load in the model before the training
# weights = "F5_4_50000_b32_lr0.01_91features_weights_epoch50.pth"
weights = "all_data_4_465000_b64_lr0.004_91features_weights_epoch75.pth"
# # Train and evaluate the model using 80% for training and 20% for testing
train_losses, test_losses = train_and_evaluate(Net91, X_data,
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


# Set print options to round to 2 decimals
np.set_printoptions(precision=2)

# Print the array
print([round(num, 3) for num in test_losses])

plot_losses(train_losses, test_losses, plrs=players,
            gms=games, tst=test_size,
            btch=batch_size, lr=learning_rate, wght=weight_decay)
