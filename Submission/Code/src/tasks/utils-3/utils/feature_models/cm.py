import numpy as np
import scipy.stats

from utils.image import Image

class ColorMoments:
    def get_window_indices(self): 
        start = 0
        stop = 64
        step = 8
        rows = [(n, min(n+step, stop)) for n in range(start, stop, step)]
        columns = rows
        return [(row, col) for row in rows for col in columns]

    def get_image_window(self, image, row_indices, column_indices):
        row_start = row_indices[0]
        row_end = row_indices[1]
        column_start = column_indices[0]
        column_end = column_indices[1]
        return image[row_start:row_end, column_start:column_end]

    def get_image_windows(self, image_matrix, window_indices):
        image_windows = []
        for indices in window_indices:
            row_indices, column_indices = indices
            image_window = self.get_image_window(image_matrix, row_indices, column_indices)
            image_windows.append(image_window)
        return image_windows

    def get_mean_fd(self, image_windows):
        means = []
        for image_window in image_windows:
            means.append(np.mean(image_window))
        return means
    
    def get_standard_deviation_fd(self, image_windows):
        standard_deviations = []
        for image_window in image_windows:
            standard_deviations.append(np.std(image_window))
        return standard_deviations

    def get_skewness_fd(self, image_windows):
        skews = []
        for image_window in image_windows:
            skew = scipy.stats.skew(image_window.flatten())
            skews.append(skew)
        return skews

    def get_color_moments_fd(self, image_matrix):
        feature_descriptor = []
        window_indices = self.get_window_indices()
        image_windows = self.get_image_windows(image_matrix, window_indices)

        mean_feature_descriptors = self.get_mean_fd(image_windows) #64
        standard_deviation_feature_descriptors = self.get_standard_deviation_fd(image_windows) #64
        skewness_feature_descriptors = self.get_skewness_fd(image_windows) #64

        feature_descriptor += mean_feature_descriptors
        feature_descriptor += standard_deviation_feature_descriptors
        feature_descriptor += skewness_feature_descriptors

        return feature_descriptor

    def compute(self, images):
        for image in images:
            image.feature_vector = self.get_color_moments_fd(image.matrix)
        
        return images

