set mode="feedback query"
set L=15
set k=100
set input_type="original feature vectors"
set transformation_matrix_file_path="..//Outputs/Task7/transformation_matrix.json"
set images_folder_path="../../../1000"
set feature_model=ELBP
set query_image_path="../../../1000/image-jitter-5-3.png"
set t=10
set output_folder_path="../Outputs/Task7"
set output_filename="results_feedback.json"
set results_file_path="../Outputs/Task7/similar_images.csv"
set dimensionality_reduction_technique=PCA
set classifier="SVM"

python src/tasks/Task8.py ^
--mode %mode% ^
--L %L% ^
--k %k% ^
--input_type %input_type% ^
--transformation_matrix_file_path %transformation_matrix_file_path% ^
--images_folder_path %images_folder_path% ^
--feature_model %feature_model% ^
--query_image_path %query_image_path% ^
--t %t% ^
--output_folder_path %output_folder_path% ^
--output_filename %output_filename% ^
--results_file_path %results_file_path% ^
--dimensionality_reduction_technique %dimensionality_reduction_technique% ^
--classifier %classifier%
