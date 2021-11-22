import numpy as np
import itertools

class Kernel:
    def declare_kernel_function(self, gamma):
        if self.type == 'linear':
            self.kernel_function = lambda x1, x2: np.dot(x1, x2)
        
        elif self.type == 'rbf':
            self.kernel_function = lambda x1, x2: np.exp(-gamma * np.dot(x1 - x2, x1 - x2))

        elif self.type == 'poly':
            self.kernel_function = lambda x1, x2: (gamma * np.dot(x1, x2) + self.r) ** self.degree

    def construct_kernel_matrix(self, training_samples, gamma):
        self.declare_kernel_function(gamma)

        # kernel_fn = lambda xi, xj: np.dot(xi, xj)
        if self.type == 'linear':
            training_samples_count = training_samples.shape[0]
            kernel_matrix = np.zeros(shape=(training_samples_count, training_samples_count))
            for i in range(training_samples_count):
                for j in range(training_samples_count):
                    kernel_matrix[i, j] = self.kernel_function(training_samples[i], training_samples[j])

            return kernel_matrix
        
        # kernel_fn = lambda x_i, x_j: np.exp(-self.gamma * np.dot(x_i - x_j, x_i - x_j))
        elif self.type == 'rbf':
            training_samples_count = training_samples.shape[0]
            kernel_matrix = np.zeros(shape=(training_samples_count, training_samples_count))
            for i in range(training_samples_count):
                for j in range(training_samples_count):
                    kernel_matrix[i, j] = self.kernel_function(training_samples[i], training_samples[j])

            return kernel_matrix
        
        # kernel_fn = lambda x_i, x_j: (self.gamma * np.dot(x_i, x_j) + r) ** deg
        elif self.type == 'poly':
            training_samples_count = training_samples.shape[0]
            kernel_matrix = np.zeros(shape=(training_samples_count, training_samples_count))
            for i in range(training_samples_count):
                for j in range(training_samples_count):
                    kernel_matrix[i, j] = self.kernel_function(training_samples[i], training_samples[j])

            return kernel_matrix

    def __init__(self, type) -> None:
        self.type = type
        self.kernel_function = None
        self.degree = 3
        self.r = 0.0

