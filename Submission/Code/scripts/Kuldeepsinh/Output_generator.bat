#!/bin/bash

set model=HOG,CM,ELBP
set k_value=5,10,20,50,100
set dimensionality_reduction_technique=PCA
set images_folder_path=E:/projects/workspace/4000/4000
set output_folder_path=E:/projects/workspace/CSE515-MWDB-Phase3/Submission/Outputs/Vectors_Task5
python ../../src/tasks/Output_Generator.py --model %model% --k %k_value% --dimensionality_reduction_technique %dimensionality_reduction_technique% --images_folder_path %images_folder_path% --output_folder_path %output_folder_path%