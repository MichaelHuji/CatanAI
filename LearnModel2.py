import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm

import matplotlib.pyplot as plt


# Define the neural network (same as before)
class Net(nn.Module):
    def __init__(self, input_size=363):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.output = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)  # Optional dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x


# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    print(f"len(train_loader) = {len(train_loader)}")
    print(f"len(test_loader) = {len(test_loader)}")
    train_losses = list()
    test_losses = list()
    num_over_half_list = list()
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model to training mode
        running_loss = 0.0
        train_accuracy = 0.0
        train_size = 0
        num_over_half = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Clear gradients from previous step
            outputs = model(inputs)  # Forward pass
            train_size += len(labels)
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights
            # TODO change the threshold to be about the median point
            thrshold = 0.5
            outputs[outputs >= thrshold] = 1
            outputs[outputs < thrshold] = 0
            # num_over_half += torch.count_nonzero(outputs)
            # loss = criterion(outputs, labels)
            train_accuracy += torch.sum(torch.abs(outputs - labels)).item()
            # running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], train accuracy: {train_accuracy/train_size}")
        train_losses.append(train_accuracy/592650)
        # num_over_half_list.append(num_over_half)
        # train_losses.append(running_loss/train_size)

        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_size = 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for inputs, labels in test_loader:
                outputs = model(inputs)
                test_size += len(labels)
                # loss = criterion(outputs, labels)
                #TODO change the threshold to be about the median point
                thrshold = 0.5
                outputs[outputs >= thrshold] = 1
                outputs[outputs < thrshold] = 0
                test_loss += torch.sum(torch.abs(outputs - labels)).item()
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {test_loss / test_size}, test size: {test_size}")
        test_losses.append(test_loss / 65850)

        torch.save(model.state_dict(), f'FvF_all_129K_363feat_model_weights_epoch{epoch}.pth')

    # print(f"num_over_half_list : {(num_over_half_list)}")
    return train_losses, test_losses

# Function to evaluate the model on the test set
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    return test_loss


# Example: Splitting the data into train and test sets, training, and evaluating
def train_and_evaluate(model_class, X, Y, test_size=0.2, epochs=10, batch_size=32, learning_rate=0.001):

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
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Train the model
    train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)

    # Evaluate the model on the test set (validation loss)
    test_loss = evaluate_model(model, test_loader, criterion)
    # print(f"Test/Validation Loss: {test_loss}")
    # torch.save(model, 'FvF_all_129K_363feat_model.pth')
    # torch.save(model.state_dict(), 'FvF_all_129K_363feat_model_weights.pth')

    return train_losses, test_losses

def load_numpy_data(filename):
    data = np.load(filename)
    X = data['X']
    Y = data['Y']
    return X, Y


def plot_losses(train_losses, validation_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    plt.title(f'FvF_all_129K_363feat, test=0.1, \nbatch=8, lr=0.001, wght_dec=1e-5')

    plt.legend()
    plt.show()

# episodes=40000
# X_data, Y_data = load_numpy_data(f"FvF_init_data_x_y_{episodes}games.npz")
# X_data, Y_data = load_numpy_data(f"RvR_init_data_x_y_{episodes}games.npz")
# X_data, Y_data = load_numpy_data(f"RvR_init_data_x_y_2VP_{episodes}games.npz")

# episodes=40000
# X_data, Y_data = load_numpy_data(f"RvR_22_{episodes}games.npz")

# episodes=9000
# X_data, Y_data = load_numpy_data(f"FvF_turn_20_{episodes}games.npz")
#
# episodes=50000
# X_data, Y_data = load_numpy_data(f"WvW_22_{episodes}games_363features.npz")


episodes=6000
X_data, Y_data = load_numpy_data(f"WvW_1_{episodes}games_363features.npz")
X_data1, Y_data1 = load_numpy_data(f"WvW_2_{episodes}games_363features.npz")
X_data2, Y_data2 = load_numpy_data(f"WvW_22_{episodes}games_363features.npz")
X_data3, Y_data3 = load_numpy_data(f"WvW_turn_10_{episodes}games_363features.npz")
X_data4, Y_data4 = load_numpy_data(f"WvW_turn_20_{episodes}games_363features.npz")
X_data5, Y_data5 = load_numpy_data(f"WvW_turn_30_{episodes}games_363features.npz")


episodes=5000
X_data6, Y_data6 = load_numpy_data(f"WvW_1_{episodes}games_363features.npz")
X_data7, Y_data7 = load_numpy_data(f"WvW_2_{episodes}games_363features.npz")
X_data8, Y_data8 = load_numpy_data(f"WvW_22_{episodes}games_363features.npz")
X_data9, Y_data9 = load_numpy_data(f"WvW_turn_10_{episodes}games_363features.npz")
X_data10, Y_data10 = load_numpy_data(f"WvW_turn_20_{episodes}games_363features.npz")
X_data11, Y_data11 = load_numpy_data(f"WvW_turn_30_{episodes}games_363features.npz")

X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4, X_data5, X_data6, X_data7, X_data8, X_data9, X_data10, X_data11), axis=0)
Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5, Y_data6, Y_data7, Y_data8, Y_data9, Y_data10, Y_data11))
print(X_data.shape)

episodes=4500
X_data5, Y_data5 = load_numpy_data(f"FvF_1_{episodes}games_363features.npz")
X_data1, Y_data1 = load_numpy_data(f"FvF_2_{episodes}games_363features.npz")
X_data2, Y_data2 = load_numpy_data(f"FvF_22_{episodes}games_363features.npz")

X_data3, Y_data3 = load_numpy_data(f"FvF_turn_10_{episodes}games_363features.npz")
X_data4, Y_data4 = load_numpy_data(f"FvF_turn_20_{episodes}games_363features.npz")

X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4, X_data5), axis=0)
Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5))
print(X_data.shape)


episodes=114000
X_data5, Y_data5 = load_numpy_data(f"FvF_1_{episodes}games_363features.npz")
X_data1, Y_data1 = load_numpy_data(f"FvF_2_{episodes}games_363features.npz")
X_data2, Y_data2 = load_numpy_data(f"FvF_22_{episodes}games_363features.npz")

X_data3, Y_data3 = load_numpy_data(f"FvF_turn_10_{episodes}games_363features.npz")
X_data4, Y_data4 = load_numpy_data(f"FvF_turn_20_{episodes}games_363features.npz")

X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4, X_data5), axis=0)
Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5))
print(X_data.shape)

Y_data[Y_data == -1] = 0

print(np.count_nonzero(Y_data))
#
#
# # Train and evaluate the model using 80% for training and 20% for testing
train_losses, test_losses = train_and_evaluate(Net, X_data, Y_data, test_size=0.1, epochs=50, batch_size=8)

plot_losses(train_losses, test_losses)
# Function to plot the training and validation loss


# model = torch.load('FvF_22VP_114K_363feat_model.pth')
#
# # Make sure to call model.eval() if you're in inference mode
# model.eval()
#
# torch.save(model.state_dict(), 'FvF_22VP_114K_363feat_model_weights.pth')



