from utils.constants import IMAGE_ID, IMAGE_TYPE
from utils.image_reader import ImageReader
from utils.feature_models.hog import HistogramOfGradients
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.cm import ColorMoments

from utils.dimensionality_reduction.kmeans import KMeans
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis

from utils.constants import *

from utils.image import Image
import numpy as np

class TaskHelper:
    def __init__(self) -> None:
        pass

    def compute_feature_vectors(self, feature_model, images):
        if feature_model == COLOR_MOMENTS:
            return ColorMoments().compute(images)
        elif feature_model == EXTENDED_LBP:
            return ExtendedLocalBinaryPattern().compute(images)
        elif feature_model == HISTOGRAM_OF_GRADIENTS:
            return HistogramOfGradients().compute(images)
        else:
            raise Exception(f"Unknown feature model - {feature_model}")

    def reduce_dimensions(self, dimensionality_reduction_technique, images, k):
        if dimensionality_reduction_technique == PRINCIPAL_COMPONENT_ANALYSIS:
            return PrincipalComponentAnalysis().compute(images, k)
        elif dimensionality_reduction_technique == SINGULAR_VALUE_DECOMPOSITION:
            return SingularValueDecomposition().compute(images, k)
        elif dimensionality_reduction_technique == LATENT_DIRICHLET_ALLOCATION:
            return LatentDirichletAllocation().compute(images, k)
        elif dimensionality_reduction_technique == KMEANS:
            return KMeans().compute(images, k)
        else:
            raise Exception(
                f"Unknown dimensionality reduction technique - {dimensionality_reduction_technique}")

    def extract_class_labels(self, images, class_type: str):
        image_reader = ImageReader()
        class_labels = [''] * len(images)

        if class_type is IMAGE_TYPE:
            for index, image in enumerate(images):
                # image_type, subject_id, image_id = image_reader.parse_image_filename(image.filename) # TODO:parallelize
                class_labels[index] = image.image_type

        elif class_type is SUBJECT_ID:
            for index, image in enumerate(images):
                # image_type, subject_id, image_id = image_reader.parse_image_filename(image.filename) # TODO:parallelize
                class_labels[index] = image.subject_id

        elif class_type is IMAGE_ID:
            for index, image in enumerate(images):
                # image_type, subject_id, image_id = image_reader.parse_image_filename(image.filename) # TODO:parallelize
                class_labels[index] = image.image_id
        else:
            raise Exception("Not a supported class type.")

        return np.array(class_labels)