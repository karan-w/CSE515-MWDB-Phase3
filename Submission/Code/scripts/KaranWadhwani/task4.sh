#!/bin/bash
# Script to run task 4 on Karan's laptop
source "/Users/karanwadhwani/Documents/ASU/Fall2021/CSE515_MWDB/Project/CSE515-MWDB-Phase3/Submission/Code/venv/bin/activate"

L=15
k=100
input_type="original feature vectors"
transformation_matrix_file_path="../../Inputs/Task4/transformation_matrix.json"
images_folder_path="../../all"
feature_model=ELBP
query_image_path="../../Inputs/Task4/image-noise01-18-2.png"
t=5
output_folder_path="../Outputs/Task4"
output_filename="c"

python3.7 src/tasks/Task4.py \
--L $L \
--k $k \
--input_type "${input_type}" \
--transformation_matrix_file_path $transformation_matrix_file_path \
--images_folder_path "${images_folder_path}" \
--feature_model $feature_model \
--query_image_path "${query_image_path}" \
--t $t \
--output_folder_path "${output_folder_path}" \
--output_filename $output_filename




# Run the task to generate transformation matrix
# generate_transformation_matrix="yes"
# images_folder_path="../../all"
# feature_model=ELBP
# dimensionality_reduction_technique="PCA"
# k=100
# output_folder_path="../Outputs/Task4"
# output_filename="transformation_matrix.json"


# python3.7 src/tasks/Task4.py \
# --generate_transformation_matrix $generate_transformation_matrix \
# --images_folder_path "${images_folder_path}" \
# --feature_model $feature_model \
# --dimensionality_reduction_technique $dimensionality_reduction_technique \
# --k $k \
# --output_folder_path "${output_folder_path}" \
# --output_filename $output_filename
