import numpy as np
from sklearn.model_selection import PredefinedSplit

__all__ = ['ConstantCVSplitter']

class ConstantCVSplitter:
    def __init__(self, train_x, train_y, train_dev_x, train_dev_y):
        if train_x.shape[0] != train_y.shape[0] or train_dev_x.shape[0] != train_dev_y.shape[0]:
            raise Exception("train x/y and train_dev x/y must be of the same size")

        self.x = np.concatenate([train_x, train_dev_x])
        self.y = np.concatenate([train_y, train_dev_y])
        self.num_train_samples = train_x.shape[0]
        self.num_train_dev_samples = train_dev_x.shape[0]

        test_fold = [-1] * self.num_train_samples + [0] * self.num_train_dev_samples
        self.splitter = PredefinedSplit(test_fold)

    def split(self):
        return self.x, self.y, self.splitter
