#!/bin/bash

set input_images_folder_path="../../all"
set feature_model=ELBP
set dimensionality_reduction_technique=PCA
set reduced_dimensions_count=5
set classification_images_folder_path="../../data"
set classifier=SVM

python3.7 src/tasks/Task1.py \
--input_images_folder_path %input_images_folder_path% \
--feature_model %feature_model% \
--dimensionality_reduction_technique %dimensionality_reduction_technique% \
--reduced_dimensions_count %reduced_dimensions_count% \
--classification_images_folder_path "%classification_images_folder_path%" \
--classifier %classifier%
