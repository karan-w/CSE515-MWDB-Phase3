from utils.feature_vector import FeatureVector
from sklearn import decomposition
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

class LatentDirichletAllocation:
    def __init__(self) -> None:
        pass

    def compute_normalized_feature_vector(self, dataset_feature_vector):
        normalized_feature_vector = dataset_feature_vector.reshape(1, len(dataset_feature_vector)*len(dataset_feature_vector[0]))
        normalized_feature_vector = preprocessing.normalize(normalized_feature_vector.reshape(1, -1), axis=1, norm='max')
        normalized_feature_vector = normalized_feature_vector.reshape(len(dataset_feature_vector), len(dataset_feature_vector[0]))
        return normalized_feature_vector

    def compute_LDA(self, dataset_feature_vector, k):
        # All the intermediate computations will be stored in the attributes dictionary 
        # so that it can be stored in the output file in the end.     
        attributes = {}
        scaler = MinMaxScaler(feature_range=(0, 1))
        attributes['dataset_feature_vector'] = dataset_feature_vector
        latent_dirichlet_allocation = decomposition.LatentDirichletAllocation(n_components=k)
        normalized_feature_vector = scaler.fit_transform(dataset_feature_vector)
        latent_dirichlet_allocation.fit(normalized_feature_vector)
        reduced_dataset_feature_vector = latent_dirichlet_allocation.transform(normalized_feature_vector)
        attributes['reduced_dataset_feature_vector'] = reduced_dataset_feature_vector
        attributes['components'] = latent_dirichlet_allocation.components_.tolist()
        return reduced_dataset_feature_vector, attributes # 400 * k

    def compute_subject_LDA(self, subjects_similarity_matrix, k):
        subjects_latent_semantics, attributes = self.compute_LDA(subjects_similarity_matrix, k)
        return subjects_latent_semantics, attributes

    def compute_type_LDA(self, types_similarity_matrix, k):
        types_latent_semantics, attributes = self.compute_LDA(types_similarity_matrix, k)
        return types_latent_semantics, attributes

    def compute(self, images, k):
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        reduced_dataset_feature_vector, attributes = self.compute_LDA(dataset_feature_vector, k)
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images, attributes

    def compute_reprojection(self, query_image, components):
        reduced_dataset_feature_vector = np.dot(query_image, components.T)
        return reduced_dataset_feature_vector
