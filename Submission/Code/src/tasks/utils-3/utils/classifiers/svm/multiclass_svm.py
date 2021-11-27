from numpy.lib.function_base import rot90
from .kernel import Kernel
from .svm import SupportVectorMachine
import numpy as np
import itertools

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
        # one vs one approach used


        # Use svm_hash_map
        # key = pair(class_label1, class_label2) where
        # class_label1 is lexicographically smaller than class_label2 
        # value = SVM

        # For image_type, 12 unique class labels
        
        unique_class_labels = np.unique(class_labels) 
        
        class_pairs = list(itertools.combinations(unique_class_labels, 2))

        for class_pair in class_pairs[:1]:
            # Select 2 classes that have not been selected before
            SVM = SupportVectorMachine(self.kernel, 0.3)

             # Select all the images that belong to these classes (400 + 400) = (800 * k)  - X

            filtered_images_reduced_feature_vector = []
            filtered_class_labels = []

            for index, class_label in enumerate(class_labels):
                if class_pair[0] in class_label or class_pair[1] in class_label:
                    filtered_images_reduced_feature_vector.append(training_samples[index]) # list of list
                    filtered_class_labels.append([class_label]) # list of list


            filtered_images_reduced_feature_vector = np.stack(filtered_images_reduced_feature_vector)
            filtered_class_labels = np.stack(filtered_class_labels)

            SVM.fit(filtered_images_reduced_feature_vector, filtered_class_labels)
            print(SVM.predict(filtered_images_reduced_feature_vector[0]))
            self.svm_hash_map[class_pair] = SVM

        # While there's no SVM for every pair of class labels

        # Initialize an SVM 
        
        self.svm_hash_map
       
        # ordering - (class_label, subject_id, image_id)

        # Select corresponding class labels = (800 * 1) - 2 possible string values

        # Encode the string class labels into + 1 or - 1 - y
            
        # Pass X and y to the initialized SVM

        # Store the mapping between the SVM and the pair of class labels used 
        # in the hash map
    
        

        pass

    def predict(self):
        pass