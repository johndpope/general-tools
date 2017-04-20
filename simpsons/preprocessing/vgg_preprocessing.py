import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['VGGPreprocessing']


class VGGPreprocessing(BaseEstimator, TransformerMixin):
    VGG_MEANS = np.array([103.939, 116.779, 123.68])/255.

    def __init__(self, backwards=False):
        self.backwards=backwards
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        if not self.backwards:
            return X - self.VGG_MEANS
        else:
            return X + self.VGG_MEANS
