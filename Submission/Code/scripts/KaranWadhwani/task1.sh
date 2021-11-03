#!/bin/bash
# Script to run task 1 on Karan's laptop

input_images_folder_path="../../all"
feature_model=ELBP
dimensionality_reduction_technique=PCA
reduced_dimensions_count=5
classification_images_folder_path="../../data"
classifier=SVM

python3.7 src/tasks/Task1.py \
--input_images_folder_path "${input_images_folder_path}" \
--feature_model $feature_model \
--dimensionality_reduction_technique $dimensionality_reduction_technique \
--reduced_dimensions_count $reduced_dimensions_count \
--classification_images_folder_path "${classification_images_folder_path}" \
--classifier $classifier