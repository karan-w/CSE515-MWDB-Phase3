set L=15
set k=100
set index_tool="LSH" 
set transformation_matrix_file_path="..//Outputs/Task7/transformation_matrix.json"
set images_folder_path="../../../1000"
set feature_model=ELBP
set query_image_path="../../../1000/image-jitter-5-3.png"
set t=10
set dimensionality_reduction_technique=PCA
set classifier="DT"

python src/tasks/Task8.py ^
--L %L% ^
--k %k% ^
--index_tool %index_tool% ^
--transformation_matrix_file_path %transformation_matrix_file_path% ^
--images_folder_path %images_folder_path% ^
--feature_model %feature_model% ^
--query_image_path %query_image_path% ^
--t %t% ^
--dimensionality_reduction_technique %dimensionality_reduction_technique% ^
--classifier %classifier%
