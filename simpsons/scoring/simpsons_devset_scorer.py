from sklearn.metrics import roc_auc_score
import numpy as np

class SimpsonsDevsetScorer:

    def __init__(self, dev_X, dev_y):
        self.dev_X = dev_X
        self.dev_y = dev_y

    def __call__(self, estimator, X, y):
        # ignores given X and y. scores on dev_X and dev_y
        num_characters = self.dev_y.shape[1]
        y_pred = estimator.predict_proba(self.dev_X)
        aucs = [ roc_auc_score(self.dev_y[:,ind], y_pred[:,ind]) for ind in range(num_characters) ]
        avg_auc = np.array(aucs).mean()

        return avg_auc
