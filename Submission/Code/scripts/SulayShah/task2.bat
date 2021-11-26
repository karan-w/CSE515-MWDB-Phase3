set training_images_folder_path="../../1000"
set feature_model=ELBP
set dimensionality_reduction_technique=PCA
set reduced_dimensions_count=10
set classification_images_folder_path="../../1000"
set classifier=DT

python src/tasks/Task2.py --training_images_folder_path %training_images_folder_path% --feature_model %feature_model% --dimensionality_reduction_technique %dimensionality_reduction_technique% --reduced_dimensions_count %reduced_dimensions_count% --classifier %classifier%