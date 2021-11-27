from sklearn import cluster
from utils.feature_vector import FeatureVector
import numpy as np

class KMeans:
    def __init__(self) -> None:
        pass

    def compute_centroids(self, dataset_feature_vector, k):
        kmeans = cluster.KMeans(init="k-means++", n_clusters=k)
        kmeans.fit(dataset_feature_vector)
        centroids = kmeans.cluster_centers_ # k * m
        return centroids

    def compute_reduced_dataset_feature_vector(self, centroids, dataset_feature_vector):
        reduced_dataset_feature_vector = []
        for centroid_index in range(centroids.shape[0]):
            centroid = centroids[centroid_index,:]
            distance_centroid = np.sum((dataset_feature_vector-centroid)**2,axis=1)
            reduced_dataset_feature_vector.append(distance_centroid)

        reduced_dataset_feature_vector = np.array(reduced_dataset_feature_vector).T
        return reduced_dataset_feature_vector

    def compute_subject_KMeans(self, subjects_similarity_matrix, k):
        # All the intermediate computations will be stored in the attributes dictionary 
        # so that it can be stored in the output file in the end.     
        attributes = {}
        attributes['subjects_similarity_matrix'] = subjects_similarity_matrix
        centroids = self.compute_centroids(subjects_similarity_matrix, k)
        attributes['centroids'] = centroids
        subjects_latent_semantics = self.compute_reduced_dataset_feature_vector(centroids, subjects_similarity_matrix)
        attributes['subjects_latent_semantics'] = subjects_latent_semantics
        return subjects_latent_semantics, attributes

    def compute_type_KMeans(self, types_similarity_matrix, k):
        # All the intermediate computations will be stored in the attributes dictionary 
        # so that it can be stored in the output file in the end.     
        attributes = {}
        attributes['types_similarity_matrix'] = types_similarity_matrix
        centroids = self.compute_centroids(types_similarity_matrix, k)
        attributes['centroids'] = centroids
        types_latent_semantics = self.compute_reduced_dataset_feature_vector(centroids, types_similarity_matrix)
        attributes['types_similarity_matrix'] = types_latent_semantics
        return types_latent_semantics, attributes

    def compute(self, images, k):
        # All the intermediate computations will be stored in the attributes dictionary 
        # so that it can be stored in the output file in the end.     
        attributes = {}
        dataset_feature_vector = FeatureVector().create_dataset_feature_vector(images)
        attributes['dataset_feature_vector'] = dataset_feature_vector
        centroids = self.compute_centroids(dataset_feature_vector, k)
        attributes['centroids'] = centroids
        reduced_dataset_feature_vector = self.compute_reduced_dataset_feature_vector(centroids, dataset_feature_vector)
        attributes['reduced_dataset_feature_vector'] = reduced_dataset_feature_vector
        images = FeatureVector().assign_images_reduced_feature_vector(images, reduced_dataset_feature_vector)
        return images, attributes
    
    def compute_reprojection(self, query_image, centroids):
        reduced_dataset_feature_vector = self.compute_reduced_dataset_feature_vector(centroids, query_image) # (1, m) * (m, k)
        return reduced_dataset_feature_vector