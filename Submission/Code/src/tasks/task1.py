import argparse
import logging

from utils.image_reader import ImageReader
from utils.constants import *

from task_helper import TaskHelper

class Task1:
    def __init__(self):
        parser = self.setup_args_parser()
        # input_images_folder_path, feature_model, dimensionality_reduction_technique, reduced_dimensions_count, classification_images_folder_path, classifier
        self.args = parser.parse_args()
    
    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--input_images_folder_path', type=str, required=True)
        parser.add_argument('--feature_model', type=str, required=True)
        parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)
        parser.add_argument('--reduced_dimensions_count', type=int, required=True)
        parser.add_argument('--classification_images_folder_path', type=str, required=True)
        parser.add_argument('--classifier', type=str, required=True)

        return parser


    def build_output(self):
        output = {}

    def save_output(self, output):
        pass

    def execute(self):
        image_reader = ImageReader()
        images = image_reader.get_all_images_in_folder(self.args.input_images_folder_path)

        task_helper = TaskHelper()
        images = task_helper.compute_feature_vectors(
            self.args.feature_model, 
            images)

        images, drt_attributes = task_helper.reduce_dimensions(
            self.args.dimensionality_reduction_technique, 
            images, 
            self.args.reduced_dimensions_count)

logger = logging.getLogger(Task1.__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

def main():
    task = Task1()
    task.execute()

if __name__ == "__main__":
    main()