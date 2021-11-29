#!/bin/bash
# Script to run task 1 on Karan's laptop
source "/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/CSE515-MWDB-Phase3/Submission/Code/venv/bin/activate"

feature_model=ELBP
dimensionality_reduction_technique=PCA
reduced_dimensions_count=20
training_images_folder_path="../../all"
test_images_folder_path="../../100"
classifier=SVM

python3.7 src/tasks/Task1.py \
--feature_model $feature_model \
--dimensionality_reduction_technique $dimensionality_reduction_technique \
--reduced_dimensions_count $reduced_dimensions_count \
--training_images_folder_path "${training_images_folder_path}" \
--test_images_folder_path "${test_images_folder_path}" \
--classifier $classifier
