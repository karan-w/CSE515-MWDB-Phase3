for FEATURE_MODEL in CM ELBP HOG
do
	for DRT in PCA SVD LDA kmeans
    do

        generate_transformation_matrix="yes"
        images_folder_path="../../all"
        feature_model=$FEATURE_MODEL
        dimensionality_reduction_technique=$DRT
        k=100
        output_folder_path="../../Inputs/Task4"
        output_filename="transformation_matrix_${FEATURE_MODEL}_${DRT}_${k}.json"

        python3.7 src/tasks/Task4.py \
        --generate_transformation_matrix $generate_transformation_matrix \
        --images_folder_path "${images_folder_path}" \
        --feature_model $feature_model \
        --dimensionality_reduction_technique $dimensionality_reduction_technique \
        --k $k \
        --output_folder_path "${output_folder_path}" \
        --output_filename $output_filename


        L=15
        k=100
        input_type="original feature vectors"
        transformation_matrix_file_path="../../Inputs/Task4/transformation_matrix_${FEATURE_MODEL}_${DRT}_${k}.json"
        images_folder_path="../../all"
        feature_model=$FEATURE_MODEL
        query_image_path="../../Inputs/Task4/image-noise01-18-2.png"
        t=10
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
    done
done