import argparse
import json
import numpy as np

from utils.indexes.lsh_index import LSHIndex

from utils.image_reader import ImageReader
from utils.constants import *

from task_helper import TaskHelper


class Task4:
    def __init__(self):
        parser = self.setup_args_parser()
        self.args = parser.parse_args()

    def setup_args_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--L', type=int, required=True)
        parser.add_argument('--k', type=int, required=True)
        parser.add_argument('--input_type', type=str, required=True)
        parser.add_argument('--transformation_matrix_file_path', type=str, required=False)
        parser.add_argument('--images_folder_path', type=str, required=True)
        parser.add_argument('--feature_model', type=str, required=True)
        parser.add_argument('--query_image_path', type=str, required=True)
        parser.add_argument('--t', type=int, required=True)

        return parser

    def read_transformation_matrix(self, file_path):
        with open(file_path, 'r') as f:
            file_contents = json.load(f)

        return file_contents["drt_attributes"]['transformation_matrix']

    
    def execute(self):
        image_reader = ImageReader()
        images_filename = image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images

        task_helper = TaskHelper()
        images_feature_vector = task_helper.compute_feature_vectors(
            self.args.feature_model, 
            images_filename)

        # Read transformation_space_matrix from the file
        transformation_matrix = self.read_transformation_matrix(self.args.transformation_matrix_file_path)
        transformation_matrix = np.array(transformation_matrix)

        lsh_index = LSHIndex(
            self.args.k,
            self.args.L,
            transformation_matrix,
            "l1"
            )

        hash_tables = lsh_index.populate_index(images_feature_vector, images_filename)
        with open('/Users/harshilgandhi/Documents/MWDB_Project/mwdb-phase3/CSE515-MWDB-Phase3/Submission/Code/output.json') as file1:
            file1.write(json.dumps(hash_tables))
#         1. Build the locality sensitive hashing index 
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