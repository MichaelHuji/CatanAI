import torch
import torch.nn as nn
from catanatron_experimental.MichaelFiles.TrainNNhelpers import combine_datasets, plot_losses
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm
from catanatron_experimental.MichaelFiles.Net import Net
from TrainNNhelpers import combine_datasets, combine_two_datasets
# DEFAULT_WEIGHT='C:/Users/micha/PycharmProjects/catanProj/catanProj/catanatron_experimental/catanatron_experimental/MichaelFiles/model_weights/'
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

        torch.save(model.state_dict(), f'model_weights/{plrs}_{gms}_b{btch}_lr{lr}_weights_epoch{epoch+1}.pth')

    return train_losses, test_losses

# Example: Splitting the data into train and test sets, training, and evaluating
def train_and_evaluate(model_class, X, Y, test_size=0.1, epochs=30,
                       plrs="NN3vNN3", gms=114000, batch_size=16, lr=0.001, weight_decay=1e-5,
                       weights=None):

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

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
    model = model_class()

    if weights != None:
        # load pre-trained weights

        # Get current working directory
        cwd = os.getcwd()

        # weight_file_name = f'NN2vNN2_47K_b16_lr005_model_weights_epoch19.pth' # is the best model we found so far
        weight_file_name = 'catanatron_experimental/catanatron_experimental/MichaelFiles/model_weights/'

        weight_file_name += weights
        # Join the directory with the file name

        file_path = os.path.join(cwd, weight_file_name)

        model.load_state_dict(torch.load(file_path))
        print('loaded weights')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # TODO: add scheduler for learning rate
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    trn_los, tst_los = train_model(model, train_loader, test_loader, criterion, optimizer, epochs,
                plrs, gms, batch_size, lr)

    return trn_los, tst_los


players = "MYVF.1evF.1e"   # choose on which player data to train the model
games = 115000                # choose on data from how many to train the model
test_size = 0.1             # choose what percent of data is used in the test set
epochs = 15                 # choose the number of epochs to train the model
batch_size = 16              # choose the size of batches used in training
learning_rate = 0.01        # choose the learning rate
weight_decay = 1e-4         # choose the weight decay
weights = None              # choose the weights to load in the model before the training
# weights = 'MYVF.2evF.25e_11004_b16_lr0.001_weights_epoch10.pth'


# X_data, Y_data = combine_datasets("MYVF.2evF.2e", 12000)
# X_data, Y_data = combine_datasets("MYVF.25evF.25e", 500)
X_data, Y_data = combine_datasets("MYVF.1evF.1e", 115000)

# X_data2, Y_data2 = combine_datasets("MYVF.25evF.25e", 6001)
# X_data, Y_data = combine_two_datasets(X_data, Y_data, X_data2, Y_data2)

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