import numpy as np
import itertools

class Kernel:
    def construct_kernel_matrix(self, training_samples, gamma):
        # kernel_fn = lambda xi, xj: np.dot(xi, xj)
        if self.type == 'linear':
            training_samples_count = training_samples.shape[0]
            kernel_matrix = np.zeros(shape=(training_samples_count, training_samples_count))
            for i in range(training_samples_count):
                for j in range(training_samples_count):
                    kernel_matrix[i, j] = np.dot(training_samples[i], training_samples[j])

            return kernel_matrix
        
        # kernel_fn = lambda x_i, x_j: np.exp(-self.gamma * np.dot(x_i - x_j, x_i - x_j))
        elif self.type == 'rbf':
            training_samples_count = training_samples.shape[0]
            kernel_matrix = np.zeros(shape=(training_samples_count, training_samples_count))
            for i in range(training_samples_count):
                for j in range(training_samples_count):
                    kernel_matrix[i, j] = np.exp(-gamma * np.dot(training_samples[i] - training_samples[j], training_samples[i] - training_samples[j]))

            return kernel_matrix
        
        # kernel_fn = lambda x_i, x_j: (self.gamma * np.dot(x_i, x_j) + r) ** deg
        elif self.type == 'poly':
            training_samples_count = training_samples.shape[0]
            kernel_matrix = np.zeros(shape=(training_samples_count, training_samples_count))
            for i in range(training_samples_count):
                for j in range(training_samples_count):
                    kernel_matrix[i, j] = (gamma * np.dot(training_samples[i], training_samples[j]) + self.r) ** self.degree

            return kernel_matrix

    def __init__(self, type) -> None:
        self.type = type
        self.degree = 3
        self.r = 0.0

