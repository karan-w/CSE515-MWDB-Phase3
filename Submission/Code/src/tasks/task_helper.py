from utils.feature_models.hog import HistogramOfGradients
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.cm import ColorMoments

from utils.dimensionality_reduction.kmeans import KMeans
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis

from utils.constants import *

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