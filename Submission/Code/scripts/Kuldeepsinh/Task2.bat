#!/bin/bash

set training_images_folder_path="E:/projects/workspace/1000/1000"
set feature_model=ELBP
set dimensionality_reduction_technique=PCA
set reduced_dimensions_count=20
set test_images_folder_path="E:/projects/workspace/100/100"
set classifier=PPR

python Task2.py --training_images_folder_path %training_images_folder_path% --feature_model %feature_model% --dimensionality_reduction_technique %dimensionality_reduction_technique% --reduced_dimensions_count %reduced_dimensions_count% --test_images_folder_path %test_images_folder_path% --classifier %classifier%