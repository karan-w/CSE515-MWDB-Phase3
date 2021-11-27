from skimage import feature

class ExtendedLocalBinaryPattern:
    def __init__(self) -> None:
        pass

    def compute(self, images):
        for image in images:
            image.feature_vector = self.get_elbp_fd(image.matrix)
            
        return images

    def get_elbp_fd(self, image_matrix):
        radius = 2
        n_points = 4 * radius
        method = "ror"
        feature_descriptor = feature.local_binary_pattern(image_matrix, n_points, radius, method)
        feature_descriptor = feature_descriptor.ravel().tolist()

        return feature_descriptor #flatten into 1D array