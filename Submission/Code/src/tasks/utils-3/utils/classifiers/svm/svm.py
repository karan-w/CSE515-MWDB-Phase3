from numpy.core.fromnumeric import transpose
from numpy.lib.arraysetops import unique
from .kernel import Kernel
import numpy as np
import itertools
import cvxopt

class SupportVectorMachine:
    def __init__(self, kernel: Kernel, regularization_parameter: float) -> None:
        self.kernel = kernel
        self.regularization_parameter = regularization_parameter

        # self.kernel = None
        # self.kernel_function = None
        # self.gamma = None
        # self.lambdas = None
        # self.X_support_vectors = None
        # self.Y_support_vectors = None
        # self.w = None
        # self.b = None

    # Assumption - class labels are a vector of two possible values +1 or -1 
    def optimize_svm_equation(self, kernel_matrix, training_samples, class_labels):
        training_samples_count = training_samples.shape[0]
        features_count = training_samples.shape[1]

        # Compute X'
        X_prime = np.multiply(training_samples, class_labels)
        X_prime_transpose = X_prime.transpose()

        # Compute H = X' mul X'T
        H = np.matmul(X_prime, X_prime_transpose)
        P = cvxopt.matrix(H)

        # Compute q - n * 1
        q = np.full((training_samples_count, 1), -1).astype(np.float64)
        q = cvxopt.matrix(q)

        # Compute G
        G = np.zeros((training_samples_count, training_samples_count))
        np.fill_diagonal(G, -1)
        G = cvxopt.matrix(G)

        # Compute h
        h = np.zeros((training_samples_count, 1))
        h = cvxopt.matrix(h)

        # Compute A
        A = class_labels.transpose()
        A = cvxopt.matrix(A)

        # Combute b
        b = np.zeros(1)
        b = cvxopt.matrix(b)

        
        try:
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            raise Exception('Could not optimize the dual formulation equation.')

        # solution is a dict of many keys. Amongst them, 
        # we are interested in "x" which represents the lambdas 
        # that we are solving for.

        lambdas = np.ravel(solution['x']) # training_samples_count * 1

        valid_support_vector = lambdas > 1e-8
        self.training_samples_support_vectors = training_samples[valid_support_vector]
        self.class_labels_support_vectors = class_labels[valid_support_vector]

        self.lambdas = lambdas[valid_support_vector]

        support_vectors_index = np.arange(len(lambdas))[valid_support_vector]

        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.class_labels_support_vectors[i]
            self.b -= np.sum(self.lambdas * self.class_labels_support_vectors * kernel_matrix[support_vectors_index[i], valid_support_vector])
        self.b /= len(self.lambdas)

        if len(self.b) == 1:
            self.b = self.b[0]

        self.w = np.zeros(features_count)
        for i in range(len(self.lambdas)):
            self.w += self.lambdas[i] * self.training_samples_support_vectors[i] * self.class_labels_support_vectors[i]
        
        print('{0:d} support vectors found out of {1:d} data points'.format(len(self.lambdas), training_samples_count))
        for i in range(len(self.lambdas)):
            print('{0:d}) X: {1}\ty: {2}\tlambda: {3:.2f}'
                    .format(i + 1, self.training_samples_support_vectors[i], self.class_labels_support_vectors[i], self.lambdas[i]))
        print('Bias of the hyper-plane: {0:.3f}'.format(self.b))
        print('Weights of the hyper-plane:', self.w)


    def fit(self, training_samples: np.ndarray, class_labels: np.ndarray) -> None:
        self.training_samples_count, self.features_count = training_samples.shape

        # Construct the kernel matrix - n * n
        kernel_matrix = self.kernel.construct_kernel_matrix(training_samples)

        # Encode the two class labels into +1 and -1 
        unique_class_labels = np.unique(class_labels)
        self.unique_class_labels = sorted(unique_class_labels)

        class_labels = np.where(class_labels == unique_class_labels[0], -1, class_labels)
        class_labels = np.where(class_labels == unique_class_labels[1], 1, class_labels)

        class_labels = class_labels.astype(np.float64)

        # Use the kernel matrix in the optimization equation
        self.optimize_svm_equation(kernel_matrix, training_samples, class_labels)

    def predict(self, test_sample: np.ndarray) -> str:
        """
        test_sample - np ndarray of (1, k)
        """
        # Project test sample onto existing model
        predicted_class = np.dot(test_sample, self.w) + self.b
        if predicted_class < 0:
            return self.unique_class_labels[0]
        else:
            return self.unique_class_labels[1]
        # Convert label from +1 or -1 to real class label