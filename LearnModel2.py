# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# import numpy as np
# from tqdm import tqdm  # Import tqdm
#
# import matplotlib.pyplot as plt
#
#
# # Define the neural network (same as before)
# class Net(nn.Module):
#     def __init__(self, input_size=363):
#         super(Net, self).__init__()
#         # self.fc1 = nn.Linear(input_size, 128)
#         # self.fc2 = nn.Linear(128, 64)
#         # self.output = nn.Linear(64, 1)
#
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 16)
#         self.output = nn.Linear(16, 1)
#
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.5)  # Optional dropout for regularization
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.sigmoid(self.output(x))
#         return x
#
#
# # Function to train the model
# def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
#     print(f"len(train_loader) = {len(train_loader)}")
#     print(f"len(test_loader) = {len(test_loader)}")
#     train_losses = list()
#     test_losses = list()
#     num_over_half_list = list()
#     for epoch in tqdm(range(epochs)):
#         model.train()  # Set model to training mode
#         running_loss = 0.0
#         train_accuracy = 0.0
#         train_size = 0
#         num_over_half = 0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()  # Clear gradients from previous step
#             outputs = model(inputs)  # Forward pass
#             train_size += len(labels)
#             loss = criterion(outputs, labels)  # Compute loss
#             loss.backward()  # Backward pass (compute gradients)
#             optimizer.step()  # Update weights
#             thrshold = 0.5
#             outputs[outputs >= thrshold] = 1
#             outputs[outputs < thrshold] = 0
#             # num_over_half += torch.count_nonzero(outputs)
#             # loss = criterion(outputs, labels)
#             train_accuracy += torch.sum(torch.abs(outputs - labels)).item()
#             # running_loss += loss.item()
#         print(f"Epoch [{epoch + 1}/{epochs}], train accuracy: {train_accuracy/train_size}")
#         train_losses.append(train_accuracy/train_size)
#         # num_over_half_list.append(num_over_half)
#         # train_losses.append(running_loss/train_size)
#
#         model.eval()  # Set model to evaluation mode
#         test_loss = 0.0
#         test_size = 0
#         with torch.no_grad():  # Disable gradient computation for evaluation
#             for inputs, labels in test_loader:
#                 outputs = model(inputs)
#                 test_size += len(labels)
#                 # loss = criterion(outputs, labels)
#                 thrshold = 0.5
#                 outputs[outputs >= thrshold] = 1
#                 outputs[outputs < thrshold] = 0
#                 test_loss += torch.sum(torch.abs(outputs - labels)).item()
#         print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {test_loss / test_size}, test size: {test_size}")
#         test_losses.append(test_loss / test_size)
#
#         torch.save(model.state_dict(), f'NN3vNN3_114K_b8_lr0002_model_weights_epoch{epoch+1}.pth')
#
#     # print(f"num_over_half_list : {(num_over_half_list)}")
#     return train_losses, test_losses
#
# # Function to evaluate the model on the test set
# def evaluate_model(model, test_loader, criterion):
#     model.eval()  # Set model to evaluation mode
#     test_loss = 0.0
#     with torch.no_grad():  # Disable gradient computation for evaluation
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#     return test_loss
#
#
#
# #
# # def load_numpy_data(filename):
# #     data = np.load(filename)
# #     X = data['X']
# #     Y = data['Y']
# #     return X, Y
# #
# #
# # def plot_losses(train_losses, validation_losses):
# #     epochs = range(1, len(train_losses) + 1)
# #     plt.plot(epochs, train_losses, label='Train Loss')
# #     plt.plot(epochs, validation_losses, label='Validation Loss')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     # plt.title('Training and Validation Loss')
# #     plt.title(f'NN3vNN3_114K, test=0.1, \nbatch=16, lr=0.0002, wght_dec=1e-5')
# #
# #     plt.legend()
# #     plt.show()
#
# # episodes=40000
# # X_data, Y_data = load_numpy_data(f"FvF_init_data_x_y_{episodes}games.npz")
# # X_data, Y_data = load_numpy_data(f"RvR_init_data_x_y_{episodes}games.npz")
# # X_data, Y_data = load_numpy_data(f"RvR_init_data_x_y_2VP_{episodes}games.npz")
#
# # episodes=40000
# # X_data, Y_data = load_numpy_data(f"RvR_22_{episodes}games.npz")
#
# # episodes=9000
# # X_data, Y_data = load_numpy_data(f"FvF_turn_20_{episodes}games.npz")
# #
# # episodes=50000
# # X_data, Y_data = load_numpy_data(f"WvW_22_{episodes}games_363features.npz")
#
# #
# # episodes=6000
# # X_data, Y_data = load_numpy_data(f"game_simulation_data/WvW_1_{episodes}games_363features.npz")
# # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/WvW_2_{episodes}games_363features.npz")
# # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/WvW_22_{episodes}games_363features.npz")
# # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/WvW_turn_10_{episodes}games_363features.npz")
# # X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/WvW_turn_20_{episodes}games_363features.npz")
# # X_data5, Y_data5 = load_numpy_data(f"game_simulation_data/WvW_turn_30_{episodes}games_363features.npz")
# #
# #
# # episodes=5000
# # X_data6, Y_data6 = load_numpy_data(f"game_simulation_data/WvW_1_{episodes}games_363features.npz")
# # X_data7, Y_data7 = load_numpy_data(f"game_simulation_data/WvW_2_{episodes}games_363features.npz")
# # X_data8, Y_data8 = load_numpy_data(f"game_simulation_data/WvW_22_{episodes}games_363features.npz")
# # X_data9, Y_data9 = load_numpy_data(f"game_simulation_data/WvW_turn_10_{episodes}games_363features.npz")
# # X_data10, Y_data10 = load_numpy_data(f"game_simulation_data/WvW_turn_20_{episodes}games_363features.npz")
# # X_data11, Y_data11 = load_numpy_data(f"game_simulation_data/WvW_turn_30_{episodes}games_363features.npz")
# #
# # X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4, X_data5, X_data6, X_data7, X_data8, X_data9, X_data10, X_data11), axis=0)
# # Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5, Y_data6, Y_data7, Y_data8, Y_data9, Y_data10, Y_data11))
# # print(X_data.shape)
# #
# # episodes=4500
# # X_data5, Y_data5 = load_numpy_data(f"game_simulation_data/FvF_1_{episodes}games_363features.npz")
# # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/FvF_2_{episodes}games_363features.npz")
# # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/FvF_22_{episodes}games_363features.npz")
# #
# # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/FvF_turn_10_{episodes}games_363features.npz")
# # X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/FvF_turn_20_{episodes}games_363features.npz")
# #
# # X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4, X_data5), axis=0)
# # Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5))
# # print(X_data.shape)
# #
# #
# # episodes=114000
# # X_data5, Y_data5 = load_numpy_data(f"game_simulation_data/FvF_1_{episodes}games_363features.npz")
# #
# # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/FvF_2_{episodes}games_363features.npz")
# # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/FvF_22_{episodes}games_363features.npz")
# #
# # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/FvF_turn_10_{episodes}games_363features.npz")
# # X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/FvF_turn_20_{episodes}games_363features.npz")
#
# # episodes=50000
# # X_data, Y_data = load_numpy_data(f"game_simulation_data/WvW_1_50000games_363features.npz")
# # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/WvW_22_50000games_363features.npz")
# # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/WvW_turn_10_50000games_363features.npz")
# # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/WvW_turn_20_50000games_363features.npz")
#
# # episodes=5000
# # X_data, Y_data = load_numpy_data(f"game_simulation_data/NN1vNN1_1_{episodes}games_363features.npz")
# # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/NN1vNN1_2_{episodes}games_363features.npz")
# # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/NN1vNN1_22_{episodes}games_363features.npz")
# # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/NN1vNN1_turn_10_{episodes}games_363features.npz")
# # X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/NN1vNN1_turn_20_{episodes}games_363features.npz")
# # X_data5, Y_data5 = load_numpy_data(f"game_simulation_data/NN1vNN1_turn_30_{episodes}games_363features.npz")
#
#
# X_data, Y_data = load_numpy_data(f"game_simulation_data/FvF_1_114000games_363features.npz")
# X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/FvF_2_114000games_363features.npz")
# X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/FvF_22_114000games_363features.npz")
# X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/FvF_turn_10_114000games_363features.npz")
# X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/FvF_turn_20_114000games_363features.npz")
#
# X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4), axis=0)
# Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4))
# #
# # episodes=42000
# # X_data0, Y_data0 = load_numpy_data(f"game_simulation_data/NN1vNN1_1_{episodes}games_363features.npz")
# # X_data1, Y_data1 = load_numpy_data(f"game_simulation_data/NN1vNN1_2_{episodes}games_363features.npz")
# # X_data2, Y_data2 = load_numpy_data(f"game_simulation_data/NN1vNN1_22_{episodes}games_363features.npz")
# # X_data3, Y_data3 = load_numpy_data(f"game_simulation_data/NN1vNN1_turn_10_{episodes}games_363features.npz")
# # X_data4, Y_data4 = load_numpy_data(f"game_simulation_data/NN1vNN1_turn_20_{episodes}games_363features.npz")
# # X_data5, Y_data5 = load_numpy_data(f"game_simulation_data/NN1vNN1_turn_30_{episodes}games_363features.npz")
# #
# # X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4, X_data5, X_data0), axis=0)
# # Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5, Y_data0))
# #
# #
#
# # X_data = np.concatenate((X_data, X_data1, X_data2, X_data3, X_data4, X_data5), axis=0)
# # Y_data = np.concatenate((Y_data, Y_data1, Y_data2, Y_data3, Y_data4, Y_data5))
# # print(X_data.shape)
#
# Y_data[Y_data == -1] = 0
#
# train_losses, test_losses = train_and_evaluate(Net, X_data, Y_data, test_size=0.25, epochs=20, batch_size=8)
#
# plot_losses(train_losses, test_losses)
#
#
#
#
