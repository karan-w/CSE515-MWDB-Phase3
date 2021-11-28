#!/bin/bash

set b=3
set model=CM
set t=10
set k_value=5
set dimensionality_reduction_technique=PCA
set latent_semantics_file=D:\MWDB\CSE515-MWDB-Phase3\Submission\Outputs\Vectors_Task5
set images_folder_path=D:\MWDB\4000
set output_folder_path=D:\MWDB\CSE515-MWDB-Phase3\Submission\Outputs\Task5
set query_image_path=D:/MWDB/test.png
python src/tasks/Task5.py --b %b% --feature_model %model% --t %t%  --k %k_value% --images_folder_path %images_folder_path%  --latent_semantics_file %latent_semantics_file% --output_folder_path %output_folder_path% --query_image_path %query_image_path% --dimensionality_reduction_technique %dimensionality_reduction_technique%