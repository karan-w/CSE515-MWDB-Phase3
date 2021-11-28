from math import gamma
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
        self.gamma = None
        self.C = 0.1

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

        P = cvxopt.matrix(np.outer(class_labels, class_labels) * kernel_matrix)       
        q = cvxopt.matrix(-np.ones(training_samples_count))
        # Compute G and h matrix according to the type of margin used
        if self.C:
            G = cvxopt.matrix(np.vstack((-np.eye(training_samples_count),
                                         np.eye(training_samples_count))))
            h = cvxopt.matrix(np.hstack((np.zeros(training_samples_count),
                                         np.ones(training_samples_count) * self.C)))
        else:
            G = cvxopt.matrix(-np.eye(training_samples_count))
            h = cvxopt.matrix(np.zeros(training_samples_count))
        A = cvxopt.matrix(class_labels.reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        # Set CVXOPT options
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 200
        
        try:
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            raise Exception('Could not optimize the dual formulation equation.')

        # solution is a dict of many keys. Amongst them, 
        # we are interested in "x" which represents the lambdas 
        # that we are solving for.

        lambdas = np.ravel(solution['x'])
        # Find indices of the support vectors, which have non-zero Lagrange multipliers, and save the support vectors
        # as instance attributes
        is_sv = lambdas > 1e-5
        self.sv_X = training_samples[is_sv]
        self.sv_y = class_labels[is_sv]
        self.lambdas = lambdas[is_sv]
        # Compute b as 1/N_s sum_i{y_i - sum_sv{lambdas_sv * y_sv * K(x_sv, x_i}}
        sv_index = np.arange(len(lambdas))[is_sv]
        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lambdas * self.sv_y * K[sv_index[i], is_sv])
        self.b /= len(self.lambdas)
        # Compute w only if the kernel is linear
        if self.kernel == 'linear':
            self.w = np.zeros(features_count)
            for i in range(len(self.lambdas)):
                self.w += self.lambdas[i] * self.sv_X[i] * self.sv_y[i]
        else:
            self.w = None
        self.is_fit = True

        if self.kernel.type == 'linear':
            self.w = np.zeros(features_count)
            for i in range(len(self.lambdas)):
                self.w += self.lambdas[i] * self.training_samples_support_vectors[i] * self.class_labels_support_vectors[i]
        else:
            self.w = None
        
        print('{0:d} support vectors found out of {1:d} data points'.format(len(self.lambdas), training_samples_count))

        for i in range(len(self.lambdas)):
            print('{0:d}) X: {1}\ty: {2}\tlambda: {3:.2f}'
                        .format(i + 1, self.sv_X[i], self.sv_y[i], self.lambdas[i]))
            print('Bias of the hyper-plane: {0:.3f}'.format(self.b))
            print('Weights of the hyper-plane:', self.w)


    def fit(self, training_samples: np.ndarray, class_labels: np.ndarray) -> None:
        self.training_samples_count, self.features_count = training_samples.shape

        # if gamma is None
        if not self.gamma:
            self.gamma = 0.01
        
        # Construct the kernel matrix - n * n
        kernel_matrix = self.kernel.construct_kernel_matrix(training_samples, self.gamma)

        # Encode the two class labels into +1 and -1 
        unique_class_labels = np.unique(class_labels)
        self.unique_class_labels = sorted(unique_class_labels)

        class_labels = np.where(class_labels == unique_class_labels[0], -1, class_labels)
        class_labels = np.where(class_labels == unique_class_labels[1], 1, class_labels)

        class_labels = class_labels.astype(np.float64)

        # Use the kernel matrix in the optimization equation
        self.optimize_svm_equation(kernel_matrix, training_samples, class_labels)

    def predict(self, test_samples: np.ndarray) -> str:
        if not self.is_fit:
            raise Exception('SVMNotFitError') 
        # If the kernel is linear and 'w' is defined, the value of f(x) is determined by
        #   f(x) = X * w + b
        if self.w is not None:
            return np.dot(test_samples, self.w) + self.b
        else:
            # Otherwise, it is determined by
            #   f(x) = sum_i{sum_sv{lambda_sv y_sv K(x_i, x_sv)}}
            y_predict = np.zeros(len(test_samples))
            for k in range(len(test_samples)):
                for lda, sv_X, sv_y in zip(self.lambdas, self.sv_X, self.sv_y):
                    # Extract the two dimensions from sv_X if 'i' and 'j' are specified
                    # if i or j:
                    #     sv_X = np.array([sv_X[i], sv_X[j]])

                    y_predict[k] += lda * sv_y * self.kernel_fn(test_samples[k], sv_X)
            return y_predict + self.b