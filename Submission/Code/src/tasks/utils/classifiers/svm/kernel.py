import numpy as np
import itertools

class Kernel:
    def construct_kernel_matrix(self, training_samples):
        if self.type is 'linear':
            training_samples_count = training_samples.shape[0]
            kernel_matrix = np.zeros(shape=(training_samples_count, training_samples_count))
            for i in range(training_samples_count):
                for j in range(training_samples_count):
                    kernel_matrix[i, j] = np.dot(training_samples[i], training_samples[j])

            kernel_fn = lambda xi, xj: np.dot(xi, xj)

            return kernel_matrix

    def __init__(self, type) -> None:
        self.type = type

