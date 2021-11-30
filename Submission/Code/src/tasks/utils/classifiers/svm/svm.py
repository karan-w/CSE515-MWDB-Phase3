from .kernel import Kernel
import numpy as np
import cvxopt

class SupportVectorMachine:
    def __init__(self, kernel: Kernel, regularization_parameter: float) -> None:
        self.kernel = kernel
        self.regularization_parameter = regularization_parameter
        self.gamma = None
        self.C = 1000

        # self.kernel = None
        # self.kernel_function = None
        # self.gamma = None
        self.lambdas = None
        self.X_support_vectors = None
        self.Y_support_vectors = None
        self.w = None
        self.b = None

    # Assumption - class labels are a vector of two possible values +1 or -1 
    def optimize_svm_equation(self, kernel_matrix, training_samples, class_labels):
        training_samples_count = training_samples.shape[0]
        features_count = training_samples.shape[1]

        P = np.outer(class_labels, class_labels) * kernel_matrix    

        # Compute q - n * 1
        q = np.full((training_samples_count,), -1).astype(np.float64)

        # Compute G
        if self.C:
            G = np.vstack((-np.eye(training_samples_count), np.eye(training_samples_count)))
            h = np.hstack((np.zeros(training_samples_count), np.ones(training_samples_count) * self.C))

        else:
            G = -np.eye(training_samples_count)
            h = np.zeros(training_samples_count)

        # Compute A
        A = class_labels.transpose()

        # Combute b
        B = np.zeros(1)
        
        # Prepare the matrices required to run the qp solver
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        B = cvxopt.matrix(B)

        try:
            solution = cvxopt.solvers.qp(P, q, G, h, A, B)
        except ValueError as e:
            raise Exception('Could not optimize the dual formulation equation.')

        # solution is a dict of many keys. Amongst them, 
        # we are interested in "x" which represents the lambdas 
        # that we are solving for.

        lambdas = np.ravel(solution['x']) # training_samples_count * 1
        valid_support_vectors = lambdas > 1e-9
        self.X_support_vectors = training_samples[valid_support_vectors]
        self.Y_support_vectors = class_labels[valid_support_vectors]
        self.lambdas = lambdas[valid_support_vectors]
        support_vectors_index = np.arange(len(lambdas))[valid_support_vectors]
        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.Y_support_vectors[i]
            self.b -= np.sum(self.lambdas * self.Y_support_vectors * kernel_matrix[support_vectors_index[i], valid_support_vectors])
        self.b /= len(self.lambdas)
        
        # threshold = 1e-6
        # valid_support_vector = lambdas > threshold

        # while(True):
        #     self.lambdas = lambdas[valid_support_vector]

        #     if(len(self.lambdas) == 0):
        #         threshold *= 1e-1
        #         valid_support_vector = lambdas > threshold
        #         continue
        #     else:
        #         self.training_samples_support_vectors = training_samples[valid_support_vector]
        #         self.class_labels_support_vectors = class_labels[valid_support_vector]

        #         support_vectors_index = np.arange(len(lambdas))[valid_support_vector]

        #         self.b = 0
        #         for i in range(len(self.lambdas)):
        #             self.b += self.class_labels_support_vectors[i]
        #             self.b -= np.sum(self.lambdas * self.class_labels_support_vectors * kernel_matrix[support_vectors_index[i], valid_support_vector])

        #         self.b /= len(self.lambdas)
        #         break

        # if len(self.b) == 1:
        #     self.b = self.b[0]

        if self.kernel.type == 'linear':
            self.w = np.zeros(features_count)
            for i in range(len(self.lambdas)):
                self.w += self.lambdas[i] * self.X_support_vectors[i] * self.Y_support_vectors[i]
        else:
            self.w = None


    def fit(self, training_samples: np.ndarray, class_labels: np.ndarray) -> None:
        self.training_samples_count, self.features_count = training_samples.shape

        # if gamma is None
        if not self.gamma:
            self.gamma = 1
        
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

    def predict(self, test_sample: np.ndarray) -> str:
        """
        test_sample - np ndarray of (1, k)

        having trouble computing predicted_class_label for non-linear kernel
        """
        # Project test sample onto existing model
        if self.w is not None:
            predicted_class_label = np.dot(test_sample, self.w) + self.b
        else:
            predicted_class_label = 0
            for k in range(len(test_sample)):
                for multipliers, support_vectors_x, support_vectors_y in zip(self.lambdas, self.X_support_vectors, self.Y_support_vectors):
                    predicted_class_label += multipliers * support_vectors_y * self.kernel.kernel_function(test_sample[k], support_vectors_x)
                predicted_class_label = predicted_class_label + self.b

        if np.sign(predicted_class_label) == -1:
            return self.unique_class_labels[0]
        else:
            return self.unique_class_labels[1]