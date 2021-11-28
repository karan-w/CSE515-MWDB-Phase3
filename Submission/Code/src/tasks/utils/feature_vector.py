from re import sub
import numpy as np

class FeatureVector:
    def __init__(self) -> None:
        pass

    def create_dataset_feature_vector(self, images):
        # images = sorted(images, key=lambda image: (image.subject_id, image.image_id))
        dataset_feature_vector = []
        for image in images:
            dataset_feature_vector.append(image.feature_vector)
        dataset_feature_vector = np.stack(dataset_feature_vector, axis=0) # 400 x f v_size
        return dataset_feature_vector

    def create_subjects_feature_vector(self, subjects):
        subjects = sorted(subjects, key=lambda subject: (subject.subject_id))
        subjects_feature_vector = []
        for subject in subjects:
            subjects_feature_vector.append(subject.feature_vector)
        subjects_feature_vector = np.stack(subjects_feature_vector, axis=0)
        return subjects_feature_vector

    def create_types_feature_vector(self, types):
        types = sorted(types, key=lambda subject: (subject.image_type))
        types_feature_vector = []
        for type in types:
            types_feature_vector.append(type.feature_vector)
        types_feature_vector = np.stack(types_feature_vector, axis=0)
        return types_feature_vector

    def create_subjects_reduced_feature_vector(self, subjects):
        subjects = sorted(subjects, key=lambda subject: (subject.subject_id))
        subjects_feature_vector = []
        for subject in subjects:
            subjects_feature_vector.append(subject.reduced_feature_vector)
        subjects_feature_vector = np.stack(subjects_feature_vector, axis=0)
        return subjects_feature_vector

    def create_types_reduced_feature_vector(self, types):
        types = sorted(types, key=lambda subject: (subject.image_type))
        types_feature_vector = []
        for type in types:
            types_feature_vector.append(type.reduced_feature_vector)
        types_feature_vector = np.stack(types_feature_vector, axis=0)
        return types_feature_vector

    
    def create_images_reduced_feature_vector(self, images):
        """
        This function takes a list of Image objects and extracts their 
        reduced dimensions feature vector and stacks them all to produce an n * k 
        numpy ndarray.
        NOTE - NO SORTING IS HAPPENING HERE. 
        """
        images_reduced_feature_vector = []
        for image in images:
            images_reduced_feature_vector.append(image.reduced_feature_vector)
        
        images_reduced_feature_vector = np.stack(images_reduced_feature_vector, axis=0)
        return images_reduced_feature_vector

    '''
    This function assumes that the images are sorted by (subject_id, and image_id) 
    and the feature_vector array follows the same sorted order. It's the responsibility 
    of the caller to provide the two inputs in sorted order. 
    '''
    def assign_images_reduced_feature_vector(self, images, reduced_dataset_feature_vector):
        for index, image in enumerate(images):
            image.reduced_feature_vector = reduced_dataset_feature_vector[index]
        return images
