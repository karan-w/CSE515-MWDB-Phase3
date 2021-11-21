import argparse
import logging

from utils.image_reader import ImageReader
from utils.constants import *


from task_helper import TaskHelper

class Task3:
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

    # def log_args(self, args):
    #     logger.debug("Received the following arguments.")
    #     logger.debug(f'model - {args.model}')
    #     # logger.debug(f'x - {args.x}')
    #     logger.debug(f'k - {args.k}')
    #     logger.debug(
    #         f'dimensionality_reduction_technique - {args.dimensionality_reduction_technique}')
    #     logger.debug(f'images_folder_path_1 - {args.images_folder_path1}')
    #     logger.debug(f'images_folder_path_2 - {args.images_folder_path2}')
    #
    #     logger.debug(f'output_folder_path - {args.output_folder_path}')

def main():
    task = Task3()
    task_helper = TaskHelper()

    parser = task.setup_args_parser()
    args = parser.parse_args()
    task.log_args(args)

    image_reader = ImageReader()

    images = image_reader.get_images_by_subjects(
        args.images_folder_path, args.x)

    images = task_helper.compute_feature_vectors(args.model, images)
    images, drt_attributes = task_helper.reduce_dimensions(
        args.dimensionality_reduction_technique, images, args.k)
    task.execute()

if __name__ == "__main__":
    main()