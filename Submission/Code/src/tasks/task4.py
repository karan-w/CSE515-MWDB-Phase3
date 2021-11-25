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


class Task4:
    def __init__(self):
        parser = self.setup_args_parser()
        self.args = parser.parse_args()

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

    def reduce_dimensions(self, dimensionality_reduction_technique, images, k):
        if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
            return PrincipalComponentAnalysis().compute(images, k)
        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            return SingularValueDecomposition().compute(images, k)
        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            return LatentDirichletAllocation().compute(images, k)
        elif dimensionality_reduction_technique == KMEANS:
            return KMeans().compute(images, k)
        else:
            raise Exception(
                f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

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
        timestamp_folder_path = Output().create_timestamp_folder(self.args.output_folder_path)
        # /Outputs/Task1/2021-10-21-23-25-23 -> /Outputs/Task1/2021-10-21-23-25-23/output.json
        output_json_path = os.path.join(
            timestamp_folder_path, 
            self.args.output_filename)
        Output().save_dict_as_json_file(output, output_json_path)

    def generate_transformation_matrix(self):
        # 1. Perform dimensionality reduction
        self.images, self.drt_attributes = self.reduce_dimensions(self.args.dimensionality_reduction_technique, self.images, self.args.k)

        # 2. Extract the transformation matrix on the basis of the drt technique used
        transformation_matrix = self.extract_transformation_matrix(self.args.dimensionality_reduction_technique, self.drt_attributes)

        # 3. Save the transformaton matrix to the output file
        output = {
            'transformation_matrix': transformation_matrix.tolist()
        }
        self.save_output(output)


    def run_task(self):
         # Read transformation_space_matrix from the file
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
        
        print("Cityblock Distance")
        for image in similar_images:
            print(image.filename)

    def execute(self):
        self.image_reader = ImageReader()
        self.images = self.image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images

        self.task_helper = TaskHelper()
        self.images = self.task_helper.compute_feature_vectors(
            self.args.feature_model, 
            self.images)

        if(self.args.generate_transformation_matrix):
            self.generate_transformation_matrix()
        else:
            self.run_task()
       
        
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

def main():
    task = Task4()
    task.execute()

if __name__ == "__main__":
    main()