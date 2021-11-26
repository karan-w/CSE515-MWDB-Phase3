import argparse
import logging

from utils.feature_vector import FeatureVector

from utils.constants import IMAGE_TYPE

# from utils.classifiers.svm.kernel import Kernel
# from utils.classifiers.svm.multiclass_svm import MultiClassSVM

from utils.classifiers.dt.dt import DecisionTreeClassifier

from utils.image_reader import ImageReader
from utils.constants import *

from task_helper import TaskHelper

from sklearn.preprocessing import LabelEncoder

"""
This class implements task 1 functionality.
"""
class Task1:
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

        # Step 5 - Extract feature vectors of all the testinng images - n' * m

        # Step 6 - Reduce the dimensions of the feature vectors of all the testing images n' * k

        # Part B - Create classifiers from the training images data
        # Step 1 - Train SVM classifier on the training images n * k 


        # linear_kernel = Kernel('linear')
        # multiclass_svm = MultiClassSVM(linear_kernel)
        # multiclass_svm.fit(training_images_reduced_feature_vectors, class_labels)


        # Step 2 - Train decision tree classifier on the training images n * k

        dt = DecisionTreeClassifier(5)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(class_labels)
        dt.fit(training_images_reduced_feature_vectors, y)
        labels = dt.predict(training_images_reduced_feature_vectors)

        print(y)
        print(labels)

        count = 0
        for i in range(len(labels)):
            if(labels[i] == y[i]):
                count += 1
        
        print(count/(len(labels)))

        # Step 3 - Train personalized page rank classifier on the training images n * k 

        # Part C - Assign labels to the test images 
        # Step 1 - Test trained SVM classifier on the testing images 
        # svm.predict() # We'll pass all test images and we'll get as output list of class labels
        # Step 2 - Test trained decision tree classifier on the testing images 

        # Step 3 - Test personalized page rank classifier on the testing images 

        # Part D - Evaluate the classifications done by each classifier on the test images 
        # Step 1 - Create confusion matrix for SVM classifier
        # confusion_matrix = ConfusionMatrix(true_labels, predicted_labels)
        # confusion_matrix.true_positive
        # confusion_matrix.true_negative
        # confusion_matrix.false_positive
        # confusion_matrix.false_negative
        # confusion_matrix.miss_rate

        # Step 2 - Compute false positives and miss rate for SVM classifier

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