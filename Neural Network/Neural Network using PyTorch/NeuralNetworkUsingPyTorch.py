import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
Preprocessing Data
"""
dataframe_train = pd.read_csv('largeTrain.csv', header = None )
dataframe_test = pd.read_csv('largeValidation.csv', header= None)
print(dataframe_train.head())
print(dataframe_test.head())

X_train = np.array(dataframe_train.drop([0], axis = 1))
y_train = np.array(dataframe_train[0])
print(X_train)
print(y_train)
X_test = np.array(dataframe_test.drop([0], axis = 1))
y_test = np.array(dataframe_test[0])
print(X_test)
print(y_test)

class MyNetwork(nn.Module):
  """
  Creating a Network
  """
  def __init__(self, inputs, neurons_hidden_layer, outputs):
    super().__init__()
    self.fc1 = nn.Linear(inputs, neurons_hidden_layer)
    self.fc2 = nn.Linear(neurons_hidden_layer, outputs)
    
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    x = torch.log_softmax(x, dim = 1)
    return x

import torch.optim as optim


"""
For trying the network at varying units and learning rates.
"""
units = [5, 20, 50, 100, 200]
rates = [0.1, 0.01, 0.001]
mega_training_loss_list = []
mega_val_loss_list = []



"""
Loop for changing number of hidden units.
"""
for neurons_hidden_layer in units:
  network = MyNetwork(X_train.shape[1], neurons_hidden_layer, 10)
  optimizer = optim.Adam(network.parameters(), lr=0.01 )

  X, y = torch.from_numpy(X_train),torch.from_numpy(y_train).long()

  X_val, y_val = torch.from_numpy(X_test),torch.from_numpy(y_test).long()
  n_epochs = 100
  X = torch.split(X, 900)
  y = torch.split(y, 900)

  train_loss_list = []
  val_loss_list = []

  for epoch in range(n_epochs):
      for batch in range(len(X)):
        network.zero_grad()
        output = network(X[batch].float())
        loss = F.cross_entropy(output, y[batch])

        output_val = network(X_val.float())
        loss_val = F.cross_entropy(output_val, y_val)


        loss.backward()
        optimizer.step()

      train_loss_list.append(loss.detach().numpy())
      val_loss_list.append(loss_val.detach().numpy())
      
  mega_training_loss_list.append(np.mean(train_loss_list))
  mega_val_loss_list.append(np.mean(val_loss_list))

plt.plot(units, mega_training_loss_list, label = "Training Loss")
plt.plot(units, mega_val_loss_list, label = "Validation Loss")
plt.title("Training and Validation Loss vs Number of Hidden Units")
plt.xlabel("Hidden Units")
plt.ylabel("Loss")
plt.legend()
plt.show()

mega_training_loss_list = []
mega_val_loss_list = []


"""
Loop for changing learning rates.
"""
for learning_rate in rates:
  network = MyNetwork(X_train.shape[1], 4, 10)
  optimizer = optim.Adam(network.parameters(), lr=learning_rate)

  X, y = torch.from_numpy(X_train),torch.from_numpy(y_train).long()

  X_val, y_val = torch.from_numpy(X_test),torch.from_numpy(y_test).long()
  n_epochs = 100
  X = torch.split(X, 900)
  y = torch.split(y, 900)


  train_loss_list = []
  val_loss_list = []

  for epoch in range(n_epochs):

      for batch in range(len(X)):
        network.zero_grad()
        output = network(X[batch].float())
        loss = F.cross_entropy(output, y[batch])

        output_val = network(X_val.float())
        loss_val = F.cross_entropy(output_val, y_val)


        loss.backward()
        optimizer.step()

      train_loss_list.append(loss.detach().numpy())
      val_loss_list.append(loss_val.detach().numpy())
      
  plt.plot(list(range(n_epochs)), train_loss_list, label = "Training Loss")
  plt.plot(list(range(n_epochs)), val_loss_list, label = "Validation Loss")
  
  plt.title("Training and Validation Loss vs Epochs at rate " + str(learning_rate))
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()
