
import numpy as np
import scipy.misc
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['ImageResizeTransformer']


class ImageResizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = np.array([scipy.misc.imresize(i, size=self.output_shape) for i in X])
        return X
