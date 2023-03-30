import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
from model import RBM
import matplotlib.pyplot as plt
import math

training_set = pd.read_csv("ml-1m/training_set.csv", delimiter=",")
training_set = np.array(training_set)

test_set = pd.read_csv("ml-1m/test_set.csv", delimiter=",")
test_set = np.array(test_set)

number_of_users = int(max(np.max(training_set[:,0]), np.max(test_set[:,0])))
number_of_movies = int(max(np.max(training_set[:,1]), np.max(test_set[:,1])))


def convert_to_matrix(data):
    new_matrix = np.zeros((number_of_users, number_of_movies), dtype='int')
    for user, movie, rating, _ in data:
        new_matrix[user-1, movie-1] = rating
    return new_matrix

training_set = convert_to_matrix(training_set)
test_set = convert_to_matrix(test_set)

training_set = torch.tensor(training_set, dtype=torch.float64)
test_set = torch.tensor(test_set, dtype=torch.float64)

# binarize ratings
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set > 2] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set > 2] = 1

rbm = RBM(number_of_movies, 200)
losses = rbm.train(training_set, 24, 20, 2)
weights = np.repeat(1.0, 10) / 10
losses_smooth = np.convolve(losses, weights, 'valid')
plt.plot(losses_smooth)

batch_size = [25,45,60]
number_of_hidden_units = [5,20,200]

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        rbm = RBM(number_of_movies, number_of_hidden_units[i])
        losses = rbm.train(training_set, batch_size[j], 20, 2)
        smoothing_factor = math.ceil(100/batch_size[j])
        weights = np.repeat(1.0, smoothing_factor) / smoothing_factor
        losses_smooth = np.convolve(losses, weights, 'valid')
        axs[i,j].plot(losses_smooth)
        axs[i,j].set_title(f'{number_of_hidden_units[i]} hidden units and batch size {batch_size[j]}')
        
fig.tight_layout()
plt.show()


    


