#!/bin/bash
# Script to run task 7 on Karan's laptop
source "/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/CSE515-MWDB-Phase3/Submission/Code/venv/bin/activate"

# mode="preliminary query"
# L=15
# k=100
# input_type="original feature vectors"
# transformation_matrix_file_path="../../Inputs/Task7/transformation_matrix.json"
# images_folder_path="../../all"
# feature_model=ELBP
# query_image_path="../../Inputs/Task7/image-noise01-18-2.png"
# t=5
# output_folder_path="../Outputs/Task7"
# output_filename="results.json"
# dimensionality_reduction_technique=PCA

# python3.7 src/tasks/Task7.py \
# --mode "$mode" \
# --L $L \
# --k $k \
# --input_type "${input_type}" \
# --transformation_matrix_file_path $transformation_matrix_file_path \
# --images_folder_path "${images_folder_path}" \
# --feature_model $feature_model \
# --dimensionality_reduction_technique $dimensionality_reduction_technique \
# --query_image_path "${query_image_path}" \
# --t $t \
# --output_folder_path "${output_folder_path}" \
# --output_filename $output_filename \
# --dimensionality_reduction_technique $dimensionality_reduction_technique


# Run the feedback query 

# source "/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/CSE515-MWDB-Phase3/Submission/Code/venv/bin/activate"

mode="feedback query"
L=15
k=100
input_type="original feature vectors"
transformation_matrix_file_path="../../Inputs/Task7/transformation_matrix.json"
images_folder_path="../../all"
feature_model=ELBP
query_image_path="../../Inputs/Task7/image-noise01-18-2.png"
t=5
output_folder_path="../Outputs/Task7"
output_filename="results_feedback.json"
results_file_path="../Outputs/Task7/similar_images.csv"
dimensionality_reduction_technique=PCA

python3.7 src/tasks/Task7.py \
--mode "${mode}" \
--L $L \
--k $k \
--input_type "${input_type}" \
--transformation_matrix_file_path $transformation_matrix_file_path \
--images_folder_path "${images_folder_path}" \
--feature_model $feature_model \
--query_image_path "${query_image_path}" \
--t $t \
--output_folder_path "${output_folder_path}" \
--output_filename $output_filename \
--results_file_path "${results_file_path}" \
--dimensionality_reduction_technique $dimensionality_reduction_technique
