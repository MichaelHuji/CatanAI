
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size=324):
        super(Net, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 16)  # Second hidden layer
        self.output = nn.Linear(16, 1)  # Output layer with a single unit (1 output)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for probability

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU to first hidden layer
        x = torch.relu(self.fc2(x))  # Apply ReLU to second hidden layer
        x = self.sigmoid(self.output(x))  # Sigmoid activation on single output
        return x


# # Create the model
# model = Net()
#
# # Print the model architecture
# print(model)
#
#
# # Define the training loop
# def train(model, X_train, Y_train, epochs, batch_size):
#     model.train()  # Set the model to training mode
#
#     # Convert training data to PyTorch tensors
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)  # Ensure Y is 2D with shape (batch_size, 1)
#
#     # Create a DataLoader to handle mini-batches
#     dataset = torch.utils.data.TensorDataset(X_train, Y_train)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     # Define the loss function and optimizer
#     criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # Training loop
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in loader:
#             # Zero the parameter gradients
#             optimizer.zero_grad()
#             # Forward pass
#             outputs = model(inputs)
#             # Compute loss
#             loss = criterion(outputs, labels)
#             # Backward pass and optimize
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader)}")
#

# # Example input (use your actual data here)
# X_train = np.random.randint(0, 100, size=(1000, 324))  # Example input
# Y_train = np.random.choice([0, 1], size=(1000,))  # Example labels (0 or 1 for each sample)
def load_numpy_data(filename):
    data = np.load(filename)
    X = data['X']
    Y = data['Y']
    return X, Y

# episodes=60000
# Load the data
# X_data, Y_data = load_numpy_data(f"FvF_init_data_x_y_{episodes}games.npz")
episodes=40000
# X_data, Y_data = load_numpy_data(f"FvF_init_data_x_y_{episodes}games.npz")
X_data, Y_data = load_numpy_data(f"RvR_init_data_x_y_{episodes}games.npz")
# print(Y_data.shape)
# split_index = len(Y_data) * 0.9
Y_data[Y_data == -1] = 0
# X_train = X_data[0:split_index, :]
# X_test = X_data[split_index:, :]
# Y_train = Y_data[0:split_index]
# Y_test = Y_data[split_index:]


#
#
# # Train the model
# train(model, X_train, Y_train, epochs=10, batch_size=32)
#
#
#
# # Switch to evaluation mode
# model.eval()
#
# with torch.no_grad():  # Disable gradient calculation for inference
#     probability = model(X_test)
#
# # print(probability)

from sklearn.model_selection import KFold

# Training function for one fold
def train_one_fold(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")


# Evaluate the model on the validation set
def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# Cross-validation function
def cross_validate(model_class, X, Y, k_folds=5, epochs=10, batch_size=32, learning_rate=0.001):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_performance = []

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split the data into training and validation sets
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        Y_train, Y_val = Y_tensor[train_idx], Y_tensor[val_idx]

        # Create data loaders for training and validation
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

        # Initialize the model and optimizer
        model = model_class()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_one_fold(model, train_loader, criterion, optimizer, epochs)

        # Evaluate the model on the validation set
        val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Validation Loss for Fold {fold + 1}: {val_loss}")
        fold_performance.append(val_loss)

    # Compute the average loss across all folds
    avg_loss = np.mean(fold_performance)
    print(f"Average Validation Loss across {k_folds} folds: {avg_loss}")

# # Perform 5-fold cross-validation
# cross_validate(Net, X_test, Y_test, k_folds=12, epochs=10, batch_size=32)
cross_validate(Net, X_data, Y_data, k_folds=10, epochs=30, batch_size=32)



