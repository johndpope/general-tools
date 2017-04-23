
import numpy as np
import scipy.misc
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['ImageResizeTransformer']


class ImageResizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, output_shape, maintain_ratio=True):
        self.output_shape = output_shape
        self.maintain_ratio = maintain_ratio

    def fit(self, X, y):
        return self

    def transform(self, X):
        should_center_image = False

        if type(self.output_shape) == float:
            size = self.output_shape
        elif self.maintain_ratio:
            size = (np.array(self.output_shape) / X.shape[1:3]).min()
            should_center_image = True
        else:
            size = self.output_shape

        X = np.array([scipy.misc.imresize(i, size=size)/255. for i in X])

        if should_center_image:
            os = list(X.shape)
            os[1:3] = self.output_shape

            X_final = np.zeros(os)
            X_final[:, (os[1]-X.shape[1])//2:(os[1]-X.shape[1])//2+X.shape[1], (os[2]-X.shape[2])//2:(os[2]-X.shape[2])//2+X.shape[2], :] = X
            X = X_final
        return X
