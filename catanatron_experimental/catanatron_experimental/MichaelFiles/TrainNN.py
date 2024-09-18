import torch
import torch.nn as nn
from catanatron_experimental.MichaelFiles.TrainNNhelpers import combine_datasets, plot_losses
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm
from catanatron_experimental.MichaelFiles.Net import Net

# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs,
                plrs="NN3vNN3", gms=114000, btch=16, lr=0.001):
    print(f"len(train_loader) = {len(train_loader)}")
    print(f"len(test_loader) = {len(test_loader)}")
    thrshold = 0.5
    train_losses = list()
    test_losses = list()
    num_over_half_list = list()
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model to training mode
        train_accuracy = 0.0
        for inputs, labels in train_loader:

            optimizer.zero_grad()  # Clear gradients from previous step
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights

            outputs[outputs >= thrshold] = 1
            outputs[outputs < thrshold] = 0
            train_accuracy += torch.sum(torch.abs(outputs - labels)).item()
        print(f"Epoch [{epoch + 1}/{epochs}], train accuracy: {train_accuracy/len(train_loader)}")
        train_losses.append(train_accuracy/len(train_loader))

        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_size = 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for inputs, labels in test_loader:
                outputs = model(inputs)

                outputs[outputs >= thrshold] = 1
                outputs[outputs < thrshold] = 0
                test_loss += torch.sum(torch.abs(outputs - labels)).item()

        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {test_loss / test_size}, test size: {test_size}")
        test_losses.append(test_loss / test_size)

        torch.save(model.state_dict(), f'{plrs}_{gms}_b{btch}_lr{lr}_weights_epoch{epoch+1}.pth')

    return train_losses, test_losses

# Example: Splitting the data into train and test sets, training, and evaluating
def train_and_evaluate(model_class, X, Y, test_size=0.1, epochs=30,
                       plrs="NN3vNN3", gms=114000, btch=16, lr=0.001,
                       weights=f'NN2vNN2_47K_b16_lr001_model_weights_epoch12.pth'):

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoaders for training and testing
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train), batch_size=btch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, Y_test), batch_size=btch, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = model_class()

    # load pre-trained weights
    model.load_state_dict(torch.load(weights))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # TODO: add scheduler for learning rate
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    trn_los, tst_los = train_model(model, train_loader, test_loader, criterion, optimizer, epochs,
                plrs, gms, btch, lr)

    return trn_los, tst_los

X_data, Y_data = combine_datasets("FvF", 114000)

# # Train and evaluate the model using 80% for training and 20% for testing
train_losses, test_losses = train_and_evaluate(Net, X_data, Y_data, test_size=0.25, epochs=20, batch_size=8)

plot_losses(train_losses, test_losses)