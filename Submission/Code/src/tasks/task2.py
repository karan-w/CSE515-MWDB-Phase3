import argparse
import logging
import numpy as np
import os
from task1_svm import SVM
from utils.classifiers.ppr_classifier import PPR
from utils.feature_vector import FeatureVector

from utils.constants import IMAGE_TYPE

from utils.classifiers.svm.kernel import Kernel
from utils.classifiers.svm.multiclass_svm import MultiClassSVM

from utils.image_reader import ImageReader
from utils.constants import *

from task_helper import TaskHelper

from sklearn.metrics import confusion_matrix

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


    def execute(self):

        image_reader = ImageReader()
        training_images = image_reader.get_all_images_in_folder(self.args.training_images_folder_path)  # 4800 images

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
        training_images = sorted(training_images,
                                 key=lambda image: (image.image_type, image.subject_id, image.image_id))

        feature_vector = FeatureVector()
        # equivalent to X in classical machine learning - np.ndarray (4800 * k)
        training_images_reduced_feature_vectors = feature_vector.create_images_reduced_feature_vector(training_images)

        # equivalent to y in classical machine learning - np.ndarray (4800 * 1)

        # class_labels = task_helper.extract_class_labels(training_images, SUBJECT_ID)
        class_labels = task_helper.extract_class_labels(training_images, SUBJECT_ID)

        # Step 4 - Read testing images from the second folder

        test_images = image_reader.get_all_images_in_folder(self.args.test_images_folder_path)  # 4800 images
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
        # true_class_labels = task_helper.extract_class_labels(test_images, SUBJECT_ID)

        true_class_labels = task_helper.extract_class_labels(test_images, SUBJECT_ID)

        # Part B - Create classifiers from the training images data
        # Step 1 - Train SVM classifier on the training images n * k

        if self.args.classifier=="PPR":
            args=dict()
            args["train_images"]=training_images
            args["test_images"] = test_images
            args["train_set_reduced_fv"] = training_images_reduced_feature_vectors
            args["test_set_reduced_fv"] = test_images_reduced_feature_vectors
            args["train_all_labels"]=class_labels
            args["test_all_labels"]=true_class_labels
            print("class_labels \n",class_labels)
            print("test_labels \n", true_class_labels)
            ppr = PPR()
            ppr.fit2(args)

        X = training_images_reduced_feature_vectors
        y = true_class_labels

def main():
    task = Task2()
    task.execute()


if __name__ == "__main__":
    main()