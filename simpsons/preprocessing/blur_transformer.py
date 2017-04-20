import PIL
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['BlurTransformer']

class BlurTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, blur_radius):
        self.blur_radius = blur_radius

    def fit(self, X, y):
        return self

    def transform(self, X):
        if (len(X.shape) == 4):
            for i in range(X.shape[0]):
                img = X[i]
                new_img = self._transform_single_image(img)
                X[i] = new_img
        elif (len(X.shape) == 3):
            X = self._transform_single_image(X)
        else:
            raise Exception("unknown input to BlurTransformer. Must be either a single image (3 dims) or a batch (4 dims). Got {} dimensions".format(len(X.shape)))
        return X


    def _transform_single_image(self, X):
        X = PIL.Image.fromarray((X*255.).astype('uint8'))
        X = X.filter(PIL.ImageFilter.GaussianBlur(radius=self.blur_radius))
        X = np.array(X)/255.
        return X
