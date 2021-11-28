import logging
import os
import sys
import argparse

from task_helper import TaskHelper
from utils.image_reader import ImageReader
from utils.output import Output
import time

sys.path.append(".")

COLOR_MOMENTS = 'CM'
EXTENDED_LBP = 'ELBP'
HISTOGRAM_OF_GRADIENTS = 'HOG'

PRINCIPAL_COMPONENT_ANALYSIS = 'PCA'
SINGULAR_VALUE_DECOMPOSITION = 'SVD'
LATENT_DIRICHLET_ALLOCATION = 'LDA'
KMEANS = 'kmeans'


class Output_Generator:
    def __init__(self):
        parser = self.setup_args_parser()
        # input_images_folder_path, feature_model, dimensionality_reduction_technique, reduced_dimensions_count, classification_images_folder_path, classifier
        self.args = parser.parse_args()


    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--k', type=str, required=True)
        parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)
        parser.add_argument('--images_folder_path', type=str, required=True)
        parser.add_argument('--output_folder_path', type=str, required=True)

        return parser

    def preprocess_drt_attributes_for_output(self, dimensionality_reduction_technique, drt_attributes):

        reduced_drt_attributes=dict()
        if (dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS):
            # dataset_feature_vector, standardized_dataset_feature_vector, eigen_values, eigen_vectors, k_principal_components_eigen_vectors

            reduced_drt_attributes['k_principal_components_eigen_vectors'] = drt_attributes[
                'k_principal_components_eigen_vectors'].real.tolist()
            reduced_drt_attributes['reduced_dataset_feature_vector'] = drt_attributes[
                'reduced_dataset_feature_vector'].real.tolist()

        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            # right_factor_matrix (V_t)
            reduced_drt_attributes['right_factor_matrix'] = drt_attributes['right_factor_matrix'].tolist(
            )
            reduced_drt_attributes['reduced_dataset_feature_vector'] = drt_attributes[
                'reduced_dataset_feature_vector'].real.tolist()

        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            # dataset_feature_vector, reduced_dataset_feature_vector
            reduced_drt_attributes['reduced_dataset_feature_vector'] = drt_attributes['reduced_dataset_feature_vector'].tolist()

        elif dimensionality_reduction_technique == KMEANS:
            # dataset_feature_vector, centroids, reduced_dataset_feature_vector
            reduced_drt_attributes['centroids'] = drt_attributes['centroids'].tolist()
            reduced_drt_attributes['reduced_dataset_feature_vector'] = drt_attributes['reduced_dataset_feature_vector'].tolist()

        return reduced_drt_attributes

    def build_output(self, drt_attributes,model,k,dimensionality_reduction_technique,images_folder_path):

        # 2. Prepare dictionary that should be JSONfied to store in JSON file
        output = {
            # args is not serializable
            'args': {
                'model': model,
                'k': k,
                'dimensionality_reduction_technique': dimensionality_reduction_technique,
                'images_folder_path': images_folder_path,
            },
            'drt_attributes': drt_attributes,
        }
        return output

    def save_output(self, output, output_folder_path,output_file_name):

        print("--------save op")
        print(type(output_folder_path))
        print(type(output_file_name))
        # output_json_path = os.path.join(
        #     output_folder_path, output_f89ile_name)
        output_json_path = output_folder_path+"/"+output_file_name
        Output().save_dict_as_json_file(output, output_json_path)


    def execute(self):

        start_time = time.time()

        task_helper = TaskHelper()
        print(self.args.model)
        models = self.args.model.split(",")
        k_vals = self.args.k.split(",")
        images_folder_path = self.args.images_folder_path
        image_reader = ImageReader()
        output_folder_path = self.args.output_folder_path
        dimensionality_reduction_technique=self.args.dimensionality_reduction_technique

        print("Reading all images")
        images = image_reader.get_all_images_in_folder(images_folder_path)
        print("all Images read")
        for model in models:
            images = task_helper.compute_feature_vectors(model, images)
            for k in k_vals:
                if k=='*':
                    k = len(images)
                k = int(k)
                print("Generating output for model ",model," and k ",k," with ",dimensionality_reduction_technique)
                images, drt_attributes = task_helper.reduce_dimensions(
                    dimensionality_reduction_technique, images, k)

                drt_attributes = self.preprocess_drt_attributes_for_output(dimensionality_reduction_technique,drt_attributes)
                output = self.build_output(drt_attributes,model,k,dimensionality_reduction_technique,images_folder_path)
                output_file_name = ""+model+"-"+str(k)+"-"+dimensionality_reduction_technique+".json"

                print(output_file_name)
                print(type(output_file_name))

                self.save_output(output,output_folder_path,output_file_name)
                print("Output saved ..... at ",(time.time()-start_time))
                print("=======================================================")

logger = logging.getLogger(Output_Generator.__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


def main():
    task = Output_Generator()
    task.execute()


if __name__ == "__main__":
    main()