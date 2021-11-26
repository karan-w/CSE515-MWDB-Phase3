import argparse
import json
import numpy as np
import os
import sys

from utils.indexes.lsh_index import LSHIndex

from utils.image_reader import ImageReader
from utils.constants import *
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.kmeans import KMeans
from utils.output import Output

from task_helper import TaskHelper

import csv

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
            'transformation_matrix': transformation_matrix.tolist()
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

    def get_similar_images(self):
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
        self.images = self.image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images

        self.task_helper = TaskHelper()
        self.images = self.task_helper.compute_feature_vectors(
            self.args.feature_model, 
            self.images)
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

    def run_task(self):
        similar_images = self.get_similar_images()
        #TODO: Save similar images to JSON file

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