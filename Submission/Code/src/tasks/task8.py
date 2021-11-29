import os

import argparse
from task_helper import TaskHelper

from task4 import Task4
from task5 import Task5

import csv
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utils.classifiers.dt.dt import DecisionTreeClassifier

from utils.image_reader import ImageReader

class Task8:
    def __init__(self):
        parser = self.setup_args_parser()
        self.args = parser.parse_args()

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--L', type=int, required=False)
        parser.add_argument('--k', type=int, required=True)
        parser.add_argument('--b', type=int, required=False)
        parser.add_argument('--latent_semantics_file', type=str, required=False)
        parser.add_argument('--index_tool', type=str,choices=['LSH','VA-File'], required=False)
        parser.add_argument('--transformation_matrix_file_path', type=str, required=False)
        parser.add_argument('--images_folder_path', type=str, required=True)
        parser.add_argument('--feature_model', type=str, required=True)
        parser.add_argument('--query_image_path', type=str, required=True)
        parser.add_argument('--t', type=int, required=True)
        parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)
        parser.add_argument("--classifier", help="What classifier to use", type=str, choices=["SVM", "DT"], required=True)

        return parser

    def plot_similar_images(self, similar_images):
        k = len(similar_images)
        # this part is for displaying 'k' similar images using matplotlib library
        rows = int(np.sqrt(k)) + 1
        cols = int(np.ceil(k/rows))
        if rows * cols <= k:
            cols += 1
        # fig = plt.figure()
        f, s_arr = plt.subplots(rows, cols)
        s_arr[0][0].axis("off")
        s_arr[0][0].text(0.5,-0.1, "Target Image", size=6, ha="center", transform=s_arr[0][0].transAxes)
        s_arr[0][0].imshow(plt.imread(self.args.query_image_path))
        i,j = 0,1
        for x in similar_images:
            if k <= 0:
                break
            s_arr[i][j].axis("off")
            s_arr[i][j].text(0.5,-0.1,x.filename, size=6, ha="center", transform=s_arr[i][j].transAxes)
            s_arr[i][j].imshow(plt.imread(os.path.join(self.args.images_folder_path, x.filename)))
            j += 1
            if j >= cols:
                j = 0
                i += 1
            k -= 1
        while i < rows:
            while j < cols:
                s_arr[i][j].axis("off")
                j += 1
            i += 1
            j = 0
        
        plt.show()

    def get_feedback(self):
        ids = input("Enter feedback - R for relevant and I for irrelevant (Comma separated) ")
        return ids.split(',')

    def run_preliminary(self, relevant_images):
        if self.args.index_tool == 'LSH':
            task4 = Task4(self.args)
            similar_images = task4.get_similar_images()
        else:
            task5 = Task5(self.args)
            similar_images = task5.get_similar_images()
        return similar_images

    def run_feedback(self, similar_images, relevant_images):
        relevant_images_filenames = []
        irrelevant_images_filenames = []
        ids = self.get_feedback()
        for i in range(len(similar_images)):
            if ids[i] == "R":
                relevant_images_filenames.append(similar_images[i].filename)
            elif ids[i] == "I":
                irrelevant_images_filenames.append(similar_images[i].filename)
            else:
                continue

        print(relevant_images_filenames)
        print(irrelevant_images_filenames)
        
        if relevant_images == None:
            image_reader = ImageReader()
            images = image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images
        else:
            images = relevant_images

        task_helper = TaskHelper()
        images = task_helper.compute_feature_vectors(
            self.args.feature_model, 
            images)

        images, drt_attributes = task_helper.reduce_dimensions(self.args.dimensionality_reduction_technique, images, self.args.k)

        # Split into training and testing
        images_hash_map = dict()
        for image in images:
            images_hash_map[image.filename] = image

        training_images = [images_hash_map[image_filename].reduced_feature_vector.real for image_filename in relevant_images_filenames] \
            + [images_hash_map[image_filename].reduced_feature_vector.real for image_filename in irrelevant_images_filenames] 

        test_images_reduced_feature_vector = []
        for image in images:
            test_images_reduced_feature_vector.append(image.reduced_feature_vector.real)
        
        training_images = np.array(training_images)
        
        if self.args.classifier == "SVM":
            class_labels = [1 for i in range(len(relevant_images_filenames))] + [-1 for j in range(len(irrelevant_images_filenames))]
            svc_model = SVC(C=10, kernel='rbf')
            svc_model.fit(training_images, class_labels)
            predicted_class_labels = svc_model.predict(test_images_reduced_feature_vector)

        elif self.args.classifier == "DT":
            class_labels = [1 for i in range(len(relevant_images_filenames))] + [0 for j in range(len(irrelevant_images_filenames))]
            dt = DecisionTreeClassifier(5)
            dt.fit(training_images, np.array(class_labels))
            predicted_class_labels = dt.predict(test_images_reduced_feature_vector)

        relevant_images_hash_map = dict()
        for class_label, image in zip(predicted_class_labels, images):
            if class_label == 1:
                relevant_images_hash_map[image.filename] = class_label
            else:
                continue

        for filename in relevant_images_filenames:
            relevant_images_hash_map[filename] = 1

        for filename in irrelevant_images_filenames:
            if filename in relevant_images_hash_map:
                relevant_images_hash_map.pop(filename)

        relevant_images = []
        
        for filename in relevant_images_hash_map:
            relevant_images.append(images_hash_map[filename])

        return relevant_images

    def execute(self):
        
        # TODO: Change generate transformation matrix thing
        self.args.output_folder_path = self.args.transformation_matrix_file_path
        self.args.output_filename = ""
        task4 = Task4(self.args)
        task4.generate_transformation_matrix()

        if self.args.index_tool == 'LSH':
            task4 = Task4(self.args)
            similar_images = task4.get_similar_images()
        else:
            task5 = Task5(self.args)
            similar_images = task5.get_similar_images()

        for image in similar_images:
            print(image.filename, image.distance_from_query_image)

        self.plot_similar_images(similar_images)
        
        feedback = feedback = str(input("Do you want to give feedback (y/n)? "))

        relevant_images = None
        while feedback == "y":

            relevant_images = self.run_feedback(similar_images, relevant_images)

            print(len(relevant_images))

            similar_images = self.run_preliminary(relevant_images)

            print("New Results: ")
            for image in similar_images:
                print(image.filename, image.distance_from_query_image)

            self.plot_similar_images(similar_images)
            
            feedback = str(input("Do you want to give feedback (y/n)? "))

def main():
    task = Task8()
    task.execute()

if __name__ == "__main__":
    main()