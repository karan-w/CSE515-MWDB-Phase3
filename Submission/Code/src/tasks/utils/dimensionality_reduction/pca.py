import re
import numpy as np

from utils.feature_vector import FeatureVector

class PrincipalComponentAnalysis:
    def __init__(self) -> None:
        pass

    def standardize_dataset_feature_vector(self, dataset_feature_vector):
        mean_dataset_feature_vector = np.mean(dataset_feature_vector, axis=0)
        std_dataset_feature_vector = np.std(dataset_feature_vector, axis=0)
        return ((dataset_feature_vector - mean_dataset_feature_vector)/std_dataset_feature_vector)

    def compute_PCA(self, dataset_feature_vector, k):
        # All the intermediate computations will be stored in the attributes dictionary 
        # so that it can be stored in the output file in the end.     
        attributes = {}
        standardized_dataset_feature_vector = self.standardize_dataset_feature_vector(dataset_feature_vector)
        attributes['dataset_feature_vector'] = dataset_feature_vector
        attributes['standardized_dataset_feature_vector'] = standardized_dataset_feature_vector
        covariance_matrix = np.cov(dataset_feature_vector, rowvar=False) # m x m

        '''
        eigen_vectors are normalized such that the column v[:,i] 
        is the eigenvector corresponding to the eigenvalue w[i]
        '''
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

        # Using ELBP and HOG as FV yields complex numbers which can't be 
        # stored in the output as complex numbers aren't serializable
        attributes['eigen_values'] = eigen_values
        attributes['eigen_vectors'] = eigen_vectors

        #sorting eigen values in descending order
        sorted_index = np.argsort(eigen_values)[::-1]

        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        k_principal_components_eigen_vectors = sorted_eigenvectors[:,0:k]
        attributes['k_principal_components_eigen_vectors'] = k_principal_components_eigen_vectors

        reduced_dataset_feature_vector = np.dot(dataset_feature_vector, k_principal_components_eigen_vectors) # (400, 1764) * (1764, 2)
        attributes['reduced_dataset_feature_vector'] = reduced_dataset_feature_vector

        return reduced_dataset_feature_vector, attributes

    def compute_subject_PCA(self, subjects_similarity_matrix, k):
        subjects_latent_semantics, attributes = self.compute_PCA(subjects_similarity_matrix, k)
        return subjects_latent_semantics, attributes

    def compute_type_PCA(self, typess_similarity_matrix, k):
        types_latent_semantics, attributes = self.compute_PCA(typess_similarity_matrix, k)
        return types_latent_semantics, attributes

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        reduced_dataset_feature_vector, attributes = self.compute_PCA(dataset_feature_vector, k)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images, attributes

    def compute_reprojection(self, query_image, k_principal_components_eigen_vectors):
        reduced_dataset_feature_vector = np.dot(query_image, k_principal_components_eigen_vectors) # (1, m) * (m, k)
        return reduced_dataset_feature_vector
