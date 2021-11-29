import argparse
import logging
import numpy as np

from utils.feature_vector import FeatureVector

from utils.constants import SUBJECT_ID

from utils.classifiers.svm.kernel import Kernel
from utils.classifiers.svm.multiclass_svm import MultiClassSVM
from utils.classifiers.dt.dt import DecisionTreeClassifier

from utils.image_reader import ImageReader
from utils.constants import *

from task_helper import TaskHelper

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

"""
This class implements task 1 functionality.
"""
class Task2:
    def __init__(self):
        parser = self.setup_args_parser()
        # input_images_folder_path, feature_model, dimensionality_reduction_technique, reduced_dimensions_count, classification_images_folder_path, classifier
        self.args = parser.parse_args()
    
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
        class_labels = task_helper.extract_class_labels(training_images, SUBJECT_ID)

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
        true_class_labels = task_helper.extract_class_labels(test_images, SUBJECT_ID)

        # Part B - Create classifiers from the training images data
        # Step 1 - Train SVM classifier on the training images n * k 
        if self.args.classifier == 'SVM':
            init_kernel = Kernel('rbf')
            multiclass_svm = MultiClassSVM(init_kernel)
            multiclass_svm.fit(training_images_reduced_feature_vectors, class_labels)
            # Step 1 - Test trained SVM classifier on the testing images 
            predicted_class_labels, votes_hash_maps = multiclass_svm.predict(test_images_reduced_feature_vectors)

        # Step 2 - Train decision tree classifier on the training images n * k 
        elif self.args.classifier == 'DT':
            label_encoder = LabelEncoder()
            class_labels = label_encoder.fit_transform(class_labels)
            true_class_labels = label_encoder.fit_transform(true_class_labels)

            dt = DecisionTreeClassifier(10)
            dt.fit(training_images_reduced_feature_vectors, class_labels)
            # Step 2 - Test trained decision tree classifier on the testing images
            predicted_class_labels = dt.predict(test_images_reduced_feature_vectors)

        # Step 3 - Train personalized page rank classifier on the training images n * k 
        elif self.args.classifier == 'PPR':
            pass

        else:
            raise Exception('Choose appropriate classification model')

        correct_predictions = 0
        wrong_predictions = 0
        for i in range(len(true_class_labels)):
            if(true_class_labels[i] == predicted_class_labels[i]):
                correct_predictions += 1
            else:
                wrong_predictions += 1

        print("correct predictions = ", correct_predictions)
        print("wrong predictions = ", wrong_predictions)

        # # Part D - Evaluate the classifications done by each classifier on the test images 
        # # Step 1 - Create confusion matrix for the classifier
        cf_matrix = confusion_matrix(true_class_labels, predicted_class_labels)
        false_positives = cf_matrix.sum(axis=0) - np.diag(cf_matrix) 
        false_negatives = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
        true_positives = np.diag(cf_matrix)
        true_negatives = cf_matrix.sum() - (false_positives + false_negatives + true_positives)

        # # Step 2 - Compute false positives and miss rate for the classifier
        miss_rate = false_negatives / (false_negatives + true_positives)
        false_positive_rate = false_positives / (false_positives + true_negatives)

        print('False positive rate: ', false_positive_rate)
        print('Miss rate: ', miss_rate)

logger = logging.getLogger(Task2.__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

def main():
    task = Task2()
    task.execute()

if __name__ == "__main__":
    main()