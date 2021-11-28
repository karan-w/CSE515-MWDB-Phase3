import argparse
import logging
import numpy as np

from task1_svm import SVM
from utils.feature_vector import FeatureVector

from utils.constants import IMAGE_TYPE

from utils.classifiers.svm.kernel import Kernel
from utils.classifiers.svm.multiclass_svm import MultiClassSVM

from utils.image_reader import ImageReader
from utils.constants import *

from task_helper import TaskHelper

from sklearn.metrics import confusion_matrix

"""
This class implements task 1 functionality.
"""
class Task1:
    def __init__(self):
        parser = self.setup_args_parser()
        # input_images_folder_path, feature_model, dimensionality_reduction_technique, reduced_dimensions_count, classification_images_folder_path, classifier
        self.args = parser.parse_args()

    def gaussian(x, z, sigma=0.1):
        return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * (sigma ** 2)))


    def setup_args_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--feature_model', type=str, required=True)
        parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)
        parser.add_argument('--reduced_dimensions_count', type=int, required=True)
        parser.add_argument('--training_images_folder_path', type=str, required=True)
        parser.add_argument('--test_images_folder_path', type=str, required=True)
        parser.add_argument('--classifier', type=str, required=True)

        return parser


    def build_output(self):
        output = {}

    def save_output(self, output):
        pass

    def execute(self):
        # Part A - Read all of the input data required for the task
        # Step 1 - Read training images from disk
        image_reader = ImageReader()
        training_images = image_reader.get_all_images_in_folder(self.args.training_images_folder_path) # 4800 images

        # Step 2 - Extract feature vectors of all the training images n * m
        task_helper = TaskHelper()
        training_images = task_helper.compute_feature_vectors(
            self.args.feature_model, 
            training_images)

        # Step 3 - Reduce the dimensions of the feature vectors of all the training images n * k
        training_images, drt_attributes = task_helper.reduce_dimensions(
            self.args.dimensionality_reduction_technique, 
            training_images, 
            self.args.reduced_dimensions_count)

        # Sort by image_type, subject_id, image_id to maintain ordering
        training_images = sorted(training_images, key=lambda image: (image.image_type, image.subject_id, image.image_id))

        feature_vector = FeatureVector()
        # equivalent to X in classical machine learning - np.ndarray (4800 * k)
        training_images_reduced_feature_vectors = feature_vector.create_images_reduced_feature_vector(training_images)

        # equivalent to y in classical machine learning - np.ndarray (4800 * 1)
        class_labels = task_helper.extract_class_labels(training_images, IMAGE_TYPE)

        # Step 4 - Read testing images from the second folder
        test_images = image_reader.get_all_images_in_folder(self.args.test_images_folder_path) # 4800 images

        # Step 5 - Extract feature vectors of all the testinng images - n' * m
        test_images = task_helper.compute_feature_vectors(
            self.args.feature_model, 
            test_images)

        # Step 6 - Reduce the dimensions of the feature vectors of all the testing images n' * k
        test_images, drt_attributes = task_helper.reduce_dimensions(
            self.args.dimensionality_reduction_technique, 
            test_images, 
            self.args.reduced_dimensions_count)

        # Sort by image_type, subject_id, image_id to maintain ordering
        test_images = sorted(test_images, key=lambda image: (image.image_type, image.subject_id, image.image_id))

        # equivalent to X in classical machine learning - np.ndarray (4800 * k)
        test_images_reduced_feature_vectors = feature_vector.create_images_reduced_feature_vector(test_images)

        # equivalent to y in classical machine learning - np.ndarray (4800 * 1)
        true_class_labels = task_helper.extract_class_labels(test_images, IMAGE_TYPE)


        # Part B - Create classifiers from the training images data
        # Step 1 - Train SVM classifier on the training images n * k 

        X = training_images_reduced_feature_vectors
        y = true_class_labels
        init_kernel = Kernel('rbf')
        svm = SVM(kernel=self.gaussian)
        svm.fit(X,y)
        y_pred = svm.predict(test_images_reduced_feature_vectors)



        # multiclass_svm = MultiClassSVM(init_kernel)
        # multiclass_svm.fit(training_images_reduced_feature_vectors, class_labels)

        # predicted_class_labels, votes_hash_maps = multiclass_svm.predict(test_images_reduced_feature_vectors)
        correct_predictions = 0
        wrong_predictions = 0
        for i in range(len(true_class_labels)):
            if(true_class_labels[i] == y_pred):
                correct_predictions += 1
            else:
                # print("true = ", true_class_labels[i])
                # print("predicted = ", predicted_class_labels[i])
                # for class_label, votes in votes_hash_maps[i].items():
                #     print(class_label, votes)
                wrong_predictions += 1

        print("correct predictions = ", correct_predictions)
        print("wrong predictions = ", wrong_predictions)

        # Step 2 - Train decision tree classifier on the training images n * k 

        # Step 3 - Train personalized page rank classifier on the training images n * k 

        # Part C - Assign labels to the test images 
        # Step 1 - Test trained SVM classifier on the testing images 
        # svm.predict() # We'll pass all test images and we'll get as output list of class labels
        # Step 2 - Test trained decision tree classifier on the testing images 

        # Step 3 - Test personalized page rank classifier on the testing images 

        # Part D - Evaluate the classifications done by each classifier on the test images 
        # Step 1 - Create confusion matrix for SVM classifier
        cf_matrix = confusion_matrix(true_class_labels, predicted_class_labels)
        false_positives = cf_matrix[0][1]
        false_negatives = cf_matrix[1][0]
        true_negatives = cf_matrix[1][1]
        true_positives = cf_matrix[0][0]

        print(false_positives, false_negatives, true_negatives, true_positives)

        # Step 2 - Compute false positives and miss rate for SVM classifier
        false_positive_rate = false_negatives / (false_negatives + true_positives)
        miss_rate = false_positives / (false_positives + true_negatives)

        print('False positive rate: ', false_positive_rate)
        print('Miss rate: ', miss_rate)
        
        # Step 3 - Create confusion matrix for decision tree classifier

        # Step 4 - Compute false positives and miss rate for decision tree classifier

        # Step 5 - Create confusion matrix for personalized page rank classifier

        # Step 6 - Compute false positives and miss rate for personalized page rank classifier

        # Part E - Store all results in the output file for task 1

logger = logging.getLogger(Task1.__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

def main():
    task = Task1()
    task.execute()

if __name__ == "__main__":
    main()