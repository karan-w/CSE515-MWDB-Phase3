import numpy as np
import json

class Subject:
    def __init__(self, subject_images) -> None:
        self.images = subject_images

        self.subject_id = subject_images[0].subject_id

        self.image_type = subject_images[0].image_type

    def create_reduced_subject_feature_vector(self, subject_images): 
        subject_images_reduced_feature_vector = np.stack([subject_image.reduced_feature_vector for subject_image in subject_images], axis=0)
        self.reduced_feature_vector = np.mean(subject_images_reduced_feature_vector, axis=0) # 1 * k

    def create_subject_feature_vector(self, subject_images):
        subject_images_feature_vector = np.stack([subject_image.feature_vector for subject_image in subject_images], axis=0) # Number of images * m
        self.feature_vector = np.mean(subject_images_feature_vector, axis=0) # 1 * m

    def create_reduced_type_feature_vector(self, type_images): 
        type_images_reduced_feature_vector = np.stack([type_image.reduced_feature_vector for type_image in type_images], axis=0)
        self.reduced_feature_vector = np.mean(type_images_reduced_feature_vector, axis=0) # 1 * k

    def create_type_feature_vector(self, type_images):
        type_images_reduced_feature_vector = np.stack([type_image.feature_vector for type_image in type_images], axis=0) # Number of images * m
        self.feature_vector = np.mean(type_images_reduced_feature_vector, axis=0) # 1 * m

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
