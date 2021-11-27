from skimage import feature

class HistogramOfGradients:
    def __init__(self) -> None:
        pass

    def compute(self, images):
        for image in images:
            image.feature_vector = self.get_hog_fd(image.matrix)
        
        return images
        
    def get_hog_fd(self, image_matrix):
        hog_feature_descriptor, hog_image = feature.hog(image_matrix, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True, 
            block_norm="L2-Hys",
            feature_vector=True)
        hog_feature_descriptor = hog_feature_descriptor.tolist()
        return hog_feature_descriptor