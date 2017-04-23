import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = [ 'BooleanPredictionSmoother' ]

class BooleanPredictionSmoother(BaseEstimator, TransformerMixin):

    def __init__(self, window_length, num_outliers):
        if window_length < num_outliers*2:
            raise Exception("window length must be at least double the number of outliers")
        self.window_length = window_length
        self.num_outliers = num_outliers


    def transform(self, preds):
        for i in range(preds.shape[0]-self.window_length+1):
            window = preds[i:i+self.window_length,:]

            for col in window.T:
                classes_count = np.bincount(col)
                classes_count_sorted = np.argsort(classes_count)
                maj_vote = classes_count_sorted[-1]
                outlier_count = classes_count[classes_count_sorted[0]]

                if outlier_count > 0 and outlier_count <= self.num_outliers \
                    and col[0]==maj_vote and col[-1]==maj_vote:
                    col[:] = maj_vote
        return preds
