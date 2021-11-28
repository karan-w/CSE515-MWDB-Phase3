import os

import argparse
from task_helper import TaskHelper

from utils.constants import FEEDBACK_QUERY, PRELIMINARY_QUERY
from task5 import Task5
from task4 import Task4

import csv
from shutil import copyfile
import numpy as np

from sklearn.svm import SVC
from utils.image_reader import ImageReader

class Task7:
    def __init__(self):
        parser = self.setup_args_parser()
        self.args = parser.parse_args()

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=str, choices=[PRELIMINARY_QUERY, FEEDBACK_QUERY], required=True)
        parser.add_argument('--index_tool', type=str,choices=['LSH','VA-File'], required=False)
        parser.add_argument('--L', type=int, required=False)
        parser.add_argument('--k', type=int, required=True)
        parser.add_argument('--b', type=int, required=True)
        parser.add_argument('--latent_semantics_file', type=str, required=True)
        parser.add_argument('--input_type', type=str, required=False)
        parser.add_argument('--transformation_matrix_file_path', type=str, required=False)
        parser.add_argument('--images_folder_path', type=str, required=True)
        parser.add_argument('--feature_model', type=str, required=True)
        parser.add_argument('--query_image_path', type=str, required=True)
        parser.add_argument('--t', type=int, required=True)
        parser.add_argument('--output_folder_path', type=str, required=True)
        parser.add_argument('--output_filename', type=str, required=False)
        parser.add_argument('--results_file_path', type=str, required=False)
        parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)

        return parser
    
    def save_similar_images(self, similar_images, feedback=False):
        # Save CSV of similar images along with distances 
        csv_filename = "similar_images.csv"
        if feedback: 
            csv_filename = "similar_images_feedback.csv"
        csv_path = os.path.join(self.args.output_folder_path, csv_filename)
        
        with open(csv_path, mode='w') as similar_images_csv_file:
            writer = csv.writer(similar_images_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Image Filename', 'Distance From Query Image', 'Feedback'])
            for similar_image in similar_images:
                row = [similar_image.filename, similar_image.distance_from_query_image]
                writer.writerow(row)

        similar_images_folder_name = "similar_images"
        if feedback:
            similar_images_folder_name = "similar_images_feedback"

        destination_folder = os.path.join(self.args.output_folder_path, similar_images_folder_name)
        try:
            os.makedirs(destination_folder)
        except:
            pass

        # Save similar images in a directory
        for similar_image in similar_images:
            source = similar_image.filepath
            destination = os.path.join(destination_folder, similar_image.filename)
            copyfile(source, destination)

    def run_preliminary_query(self):
        # We use task 4 to run the preliminary query
        if self.args.index_tool == 'LSH':
            task4 = Task4(self.args)
            similar_images = task4.get_similar_images()
        else:
            task5 = Task5(self.args)
            similar_images = task5.get_similar_images()
        self.save_similar_images(similar_images)

    def run_feedback_query(self):
        # Read the csv file and get the list of relevant and irrelevant images
        relevant_images_filenames = []
        irrelevant_images_filenames = []

        with open(self.args.results_file_path) as results_csv_file:
            csv_reader = csv.reader(results_csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    image_filename = row[0]
                    label = row[2]

                    if label == 'R':
                        relevant_images_filenames.append(image_filename)
                    elif label == 'I':
                        irrelevant_images_filenames.append(image_filename)
                    else:
                        continue
        
        image_reader = ImageReader()
        images = image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images

        task_helper = TaskHelper()
        images = task_helper.compute_feature_vectors(
            self.args.feature_model, 
            images)

        images, drt_attributes = task_helper.reduce_dimensions(self.args.dimensionality_reduction_technique, images, self.args.k)

        # Split into training and testing
        images_hash_map = dict()
        for image in images:
            images_hash_map[image.filename] = image

        training_images = [images_hash_map[image_filename].reduced_feature_vector for image_filename in relevant_images_filenames] \
            + [images_hash_map[image_filename].reduced_feature_vector for image_filename in irrelevant_images_filenames] 

        test_images_reduced_feature_vector = []
        for image in images:
            test_images_reduced_feature_vector.append(image.reduced_feature_vector)
        
        training_images = np.array(training_images)
        class_labels = [1 for i in range(len(relevant_images_filenames))] + [-1 for j in range(len(irrelevant_images_filenames))]

        svc_model = SVC(C=10, kernel='rbf')
        svc_model.fit(training_images, class_labels)

        predicted_class_labels = svc_model.predict(test_images_reduced_feature_vector)
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
            
        if self.args.index_tool == 'LSH':
            task4 = Task4(self.args)
            similar_images = task4.get_similar_images(relevant_images)
        else:
            task5 = Task5(self.args)
            similar_images = task5.get_similar_images(relevant_images)
        self.save_similar_images(similar_images, feedback=True)


    def execute(self):
        if self.args.mode == PRELIMINARY_QUERY:
            self.run_preliminary_query()
        elif self.args.mode == FEEDBACK_QUERY:
            self.run_feedback_query()

def main():
    task = Task7()
    task.execute()

if __name__ == "__main__":
    main()