from numpy.lib.function_base import rot90
from .kernel import Kernel
from .svm_test import SupportVectorMachine
import numpy as np
import itertools
import collections

class MultiClassSVM:
    def __init__(self, kernel: Kernel) -> None:
        """
        training_samples - reduced images feature vector (4800 * k)
        class_labels - (4800 * 1) - 12 unique strings 
        """
        self.svm_hash_map = dict()
        self.kernel = kernel

        pass

    def fit(self, training_samples: np.ndarray, class_labels: np.ndarray):
        """
            training_samples - numpy ndarray of size 4800 * k 
            class_labels - numpy ndarray of 4800 * 1
        """
        SVM = SupportVectorMachine(self.kernel, 0.3)
        labels = np.unique(class_labels)
        for label in labels:
            if type(label) == int:
                raise ValueError(str(label) + " is not an integer value label")
        self.labels = np.array(labels, dtype=int)

        # re-arrange training set per labels in a dictionary
        X_arranged_list = collections.defaultdict(list)
        for index, x in enumerate(training_samples):
            X_arranged_list[class_labels[index]].append(x)

        # convert to numpy array the previous dictionary
        X_arranged_numpy = {}
        for index in range(len(self.labels)):
            X_arranged_numpy[index] = np.array(X_arranged_list[index])

        for i in range(0, self.labels.shape[0] - 1):
            for j in range(i + 1, self.labels.shape[0]):
                current_X = np.concatenate((X_arranged_numpy[i], X_arranged_numpy[j]))
                current_y = np.concatenate((- np.ones((len(X_arranged_numpy[i]),), dtype=int),
                                           np.ones(len((X_arranged_numpy[j]),), dtype=int)))
                svm = SVM(kernel=self.kernel, gamma=self.gamma, deg=self.deg, r=self.r, C=self.C)
                svm.fit(current_X, current_y, verbosity=0)
                for sv in svm.sv_X:
                    self.support_vectors.add(tuple(sv.tolist()))
                svm_tuple = (svm, self.labels[i], self.labels[j])
                self.SVMs.append(svm_tuple)
        print('{0:d} support vectors found out of {1:d} data points'.format(len(self.support_vectors), len(training_samples)))

    def predict(self, testing_samples: np.ndarray):
        voting_schema = np.zeros([len(testing_samples), 2, self.labels.shape[0]], dtype=float)
        for svm_tuple in self.SVMs:
            prediction = svm_tuple[0].project(testing_samples)
            for i in range(len(prediction)):
                if prediction[i] < 0:
                    voting_schema[i][0][svm_tuple[1]] += 1
                    voting_schema[i][1][svm_tuple[1]] += -1 * prediction[i]
                else:
                    voting_schema[i][0][svm_tuple[2]] += 1
                    voting_schema[i][1][svm_tuple[2]] += prediction[i]

        voting_results = np.zeros(len(voting_schema), dtype=int)
        for i in range(len(voting_schema)):
            sorted_votes = np.sort(voting_schema[i][0])
            # if the first two classes received a different number of votes there is no draw
            if sorted_votes[0] > sorted_votes[1]:
                voting_results[i] = voting_schema[i][0].argmax()
            # otherwise return label of the class which has highest cumulative sum of predicted values
            else:
                voting_results[i] = voting_schema[i][1].argmax()

        return voting_results
        
        return predicted_class_labels, votes_hash_maps


                

