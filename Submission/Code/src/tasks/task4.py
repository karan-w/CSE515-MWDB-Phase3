import argparse
import json
import numpy as np
import os
import sys
import cv2

from utils.indexes.lsh_index import LSHIndex

from utils.image_reader import ImageReader
from utils.constants import *
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.kmeans import KMeans
from utils.output import Output
from utils.image import Image

from task_helper import TaskHelper

import csv
from shutil import copyfile

class Task4:
    def __init__(self, args = None):
        if args is None:
            parser = self.setup_args_parser()
            self.args = parser.parse_args()
        else:
            self.args = args

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--L', type=int, required=False)
        parser.add_argument('--k', type=int, required=False)
        parser.add_argument('--input_type', type=str, required=False)
        parser.add_argument('--transformation_matrix_file_path', type=str, required=False)
        parser.add_argument('--images_folder_path', type=str, required=False)
        parser.add_argument('--feature_model', type=str, required=False)
        parser.add_argument('--query_image_path', type=str, required=False)
        parser.add_argument('--t', type=int, required=False)
        parser.add_argument('--output_folder_path', type=str, required=False)
        parser.add_argument('--output_filename', type=str, required=False)
        parser.add_argument('--generate_transformation_matrix', type=str, required=False)
        parser.add_argument('--dimensionality_reduction_technique', type=str, required=False)
        parser.add_argument('--query_image_type', type=str, required=False)
        parser.add_argument('--query_image_subject_id', type=str, required=False)
        parser.add_argument('--query_image_id', type=str, required=False)

        return parser

    def read_transformation_matrix(self, file_path):
        with open(file_path, 'r') as f:
            file_contents = json.load(f)

        return file_contents['transformation_matrix']


    # def create_image_filenames_list(self, images): 
    #     image_filenames = []
    #     for image in images:
    #         image_filenames.append(image.filename)

    #     return image_filenames

    def extract_transformation_matrix(self, dimensionality_reduction_technique, attributes):
        transformation_matrix = None
        if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
            transformation_matrix = np.array(
                attributes['k_principal_components_eigen_vectors'])
        elif dimensionality_reduction_technique == KMEANS:
            transformation_matrix = np.array(attributes['centroids'])
        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            transformation_matrix = np.array(attributes['components'])
        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            transformation_matrix = np.array(attributes['right_factor_matrix'])
        
        return transformation_matrix

    def save_output(self, output):
        # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23
        # timestamp_folder_path = Output().create_timestamp_folder(self.args.output_folder_path)
        # /Outputs/Task1/2021-10-21-23-25-23 -> /Outputs/Task1/2021-10-21-23-25-23/output.json
        if self.args.output_filename == "":
            output_json_path = self.args.output_folder_path
        else:
            output_json_path = os.path.join(
                self.args.output_folder_path, 
                self.args.output_filename)
        Output().save_dict_as_json_file(output, output_json_path)

    def generate_transformation_matrix(self):
        self.image_reader = ImageReader()
        self.images = self.image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images

        self.task_helper = TaskHelper()
        self.images = self.task_helper.compute_feature_vectors(
            self.args.feature_model, 
            self.images)

        # 1. Perform dimensionality reduction
        self.images, self.drt_attributes = self.task_helper.reduce_dimensions(self.args.dimensionality_reduction_technique, self.images, self.args.k)

        # 2. Extract the transformation matrix on the basis of the drt technique used
        transformation_matrix = self.extract_transformation_matrix(self.args.dimensionality_reduction_technique, self.drt_attributes)

        # 3. Save the transformaton matrix to the output file
        output = {
            'transformation_matrix': transformation_matrix.real.tolist()
        }
        self.save_output(output)


    def evaluate(self, similar_images):
        true_image_type = "noise01"
        true_subject_id = 18
        true_image_id = 2

        relevant = 0
        non_relevant = 0

        relevant_images = []
        non_relevant_images = []

        for similar_image in similar_images:
            image_type, subject_id, image_id = self.image_reader.parse_image_filename(similar_image.filename)
            if(image_type == true_image_type or str(subject_id) == str(true_subject_id) or str(image_id) == str(true_image_id)):
                relevant += 1
                relevant_images.append(similar_image)
            else:
                non_relevant += 1
                non_relevant_images.append(similar_image)

        similarity_score = relevant/len(similar_images)

        with open('../Outputs/Task4/experiment.csv', mode='a') as experiment_csv_file:
            experiment_writer = csv.writer(experiment_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Query Image	Feature Model	Dimensionality Reduction Technique	Reduced Dimensions Count	Similar Images Count (t)	Relevant Image Filenames	Non Relevant Image Filenames	Similarity Relevance Score
            
            row = []
            row.append(self.args.query_image_path.split(os.path.sep)[-1])
            row.append(self.args.feature_model)
            row.append(self.args.transformation_matrix_file_path.split("_")[-2])
            row.append(self.args.k)
            row.append(self.args.t)
            row.append([relevant_image.filename for relevant_image in relevant_images])
            row.append([non_relevant_image.filename for non_relevant_image in non_relevant_images])
            row.append(similarity_score * 100)

            experiment_writer.writerow(row)

    def save_similar_images(self, similar_images):
        # Save CSV of similar images along with distances 
        csv_filename = self.args.output_filename
        csv_path = os.path.join(self.args.output_folder_path, csv_filename)
        
        with open(csv_path, mode='w') as similar_images_csv_file:
            writer = csv.writer(similar_images_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Image Filename', 'Distance From Query Image', 'Feedback'])
            for similar_image in similar_images:
                row = [similar_image.filename, similar_image.distance_from_query_image]
                writer.writerow(row)

        similar_images_folder_name = "similar_images"

        destination_folder = os.path.join(self.args.output_folder_path, similar_images_folder_name)
        os.makedirs(destination_folder)

        # Save similar images in a directory
        for similar_image in similar_images:
            source = similar_image.filepath
            destination = os.path.join(destination_folder, similar_image.filename)
            copyfile(source, destination)

    def get_similar_images(self, images = None):
                # 1. Build the locality sensitive hashing index 
        #     LSHI(L, k, transformation_space) [Assumption: L = transformation_space.shape[1]]
        #     1. a. Set up the hash functions from the transformation space (g1, ..., gL)
        #     1. b. Initialize empty L hash tables  (HT1, ..., HTL)

        # 2. Populate the index from the images read from the folder
        #     2. a. Obtain feature vectors for all the input images based on the feature model provided as input
        #     2. b. Insert every image into the LSHI
        #         For i = 1 ... L
        #             Compute gi(image) and store the image filename in bucket gi(image) in the ith hash table

        # 3. Find t similar images for query image q
        #         retrieved_images = []
        #     3. a. For i = 1 ... L
        #             Compute gi(query_image) and retrieve all the images located in the bucket gi(query_image) in the ith hash table (append to retrieved_images)

        #     3. b. Compute distances between query image and retrieved_images

        #     3. c. Pick the t closest images from the retrieved images

        # Read transformation_space_matrix from the file
        self.image_reader = ImageReader()
        self.task_helper = TaskHelper()

        if images is None:
            self.images = self.image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images

            self.images = self.task_helper.compute_feature_vectors(
                self.args.feature_model, 
                self.images)

        else:
            self.images = images

        transformation_matrix = self.read_transformation_matrix(self.args.transformation_matrix_file_path)
        transformation_matrix = np.array(transformation_matrix)

        lsh_index = LSHIndex(
            self.args.k,
            self.args.L,
            transformation_matrix,
            "l1",
            0.1,
            "cityblock"
            )

        # image_filenames = self.create_image_filenames_list(images)

        lsh_index.populate_index(self.images)

        print(sys.getsizeof(lsh_index))

        query_image = self.image_reader.get_query_image(self.args.query_image_path)
        query_image_feature_vector = self.task_helper.compute_query_feature_vector(self.args.feature_model, query_image)

        similar_images = lsh_index.get_similar_images(query_image_feature_vector, self.args.t, self.images)
        
        return similar_images

    def same_subject(self, image1, image2):
        return str(image1.subject_id) == str(image2.subject_id)

    def same_image_type(self, image1, image2):
        return str(image1.image_type) == str(image2.image_type)

    
    def evaluate_similar_images(self, similar_images, images):
        query_image_matrix = cv2.imread(self.args.query_image_path, cv2.IMREAD_GRAYSCALE)
        if query_image_matrix is None:
            raise Exception(f"Could not read image with the filepath {self.args.query_image_path}")

        true_image_type = None
        true_subject_id = None
        true_image_id = None

        query_image_filename = self.args.query_image_path

        if self.args.query_image_type and self.args.query_image_subject_id and self.args.query_image_id:
            true_image_type = self.args.query_image_type
            true_subject_id = self.args.query_image_subject_id
            true_image_id = self.args.query_image_id
        else:
            query_image_filename = os.path.basename(self.args.query_image_path)
            true_image_type, true_subject_id, true_image_id = self.image_reader.parse_image_filename(query_image_filename)

        query_image = Image(query_image_filename, query_image_matrix, true_subject_id, true_image_id, true_image_type, self.args.query_image_path)

        # We get the true positives and the false positives from similar_images set
        true_positives = 0
        false_positives = 0

        for similar_image in similar_images:
            if(self.same_subject(similar_image, query_image) and self.same_image_type(similar_image, query_image)):
                true_positives += 1
            else:
                false_positives += 1

        # We get the true negatives and false negatives from images - similar_images set 
        true_negatives = 0
        false_negatives = 0

        images_hash_map = dict()
        for image in images:
            images_hash_map[image.filename] = image

        for similar_image in similar_images:
            if similar_image.filename in images_hash_map:
                images_hash_map.pop(similar_image.filename)

        for image in images_hash_map.values():
            if(self.same_subject(similar_image, query_image) and self.same_image_type(similar_image, query_image)):
                false_negatives += 1
            else:
                true_negatives += 1

        miss_rate = false_negatives/(false_negatives + true_positives)
        print("False Positives = ", false_positives)
        print("Miss Rate = ", miss_rate)


    def run_task(self):
        similar_images = self.get_similar_images()
        self.save_similar_images(similar_images)

    def execute(self):
        if(self.args.generate_transformation_matrix):
            self.generate_transformation_matrix()
        else:
            self.run_task()

def main():
    task = Task4()
    task.execute()

if __name__ == "__main__":
    main()