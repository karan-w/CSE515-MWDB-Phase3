#!/bin/bash

set mode="preliminary query"
set model=CM
set t=10
set k_value=7
set dimensionality_reduction_technique=PCA
set latent_semantics_file=D:\MWDB
set images_folder_path=D:\MWDB\4000
set output_folder_path=D:\MWDB\CSE515-MWDB-Phase3\Submission\Outputs\Task5
set query_image_path=D:/MWDB/test.png
set generate_va_file = False
python src/tasks/Task7.py --mode %mode% --results_file_path D:\MWDB --feature_model %model% --t %t%  --k %k_value%  --output_folder_path %output_folder_path% --query_image_path %query_image_path%  --dimensionality_reduction_technique PCA --b 3 --images_folder_path %images_folder_path%