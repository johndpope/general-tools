from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

__all__ = ['ClassifierThreshOptimizer']


class ClassifierThresholdOptimizer:
    def __init__(self, model, dev_X, dev_y, preds=None):
        self.model = model
        self.dev_X = dev_X
        self.dev_y = dev_y
        if preds is not None:
            self.dev_preds = preds
        else:
            self.dev_preds = self.model.predict(self.dev_X)

    def fit(self, precision=None, recall=None, maximize_accuracy=False, maximize_fscore=False):
        if precision is None and recall is None and not maximize_accuracy and not maximize_fscore:
            raise Exception("either precision, recall, maximize_accuracy or maximize_fscore must be specified ")

        char_threshs = []
        for char_id in range(self.dev_y.shape[1]):
            y_true, y_pred = self.dev_y[:,char_id], self.dev_preds[:,char_id]

            if precision:
                t = self._find_thresh_for_precision(y_true, y_pred, precision)
            elif recall:
                t = self._find_thresh_for_recall(y_true, y_pred, recall)
            elif maximize_accuracy:
                t = self._find_thresh_for_max_acc(y_true, y_pred)
            elif maximize_fscore:
                t = self._find_thresh_for_max_fscore(y_true, y_pred)

            char_threshs.append(t)

        self.thresholds = np.array(char_threshs)



    def _find_thresh_for_precision(self, y_true, y_pred, precision):
        p,r,t = precision_recall_curve(y_true, y_pred)
        t = np.append(t,[1.])
        return t[p>=precision].min()

    def _find_thresh_for_recall(self, y_true, y_pred, recall):
        p,r,t = precision_recall_curve(y_true, y_pred)
        t = np.append(t,[1.])
        return t[r>=recall].max()

    def _find_thresh_for_max_acc(self, y_true, y_pred):
        accs = [ (t, (y_true == (y_pred > t).astype(int)).mean()) for t in np.linspace(0.,1.,21) ]
        return sorted(accs, key=lambda x:x[1])[-1][0]

    def _find_thresh_for_max_fscore(self, y_true, y_pred):
        fscores = [ (t, f1_score(y_true, (y_pred > t))) for t in np.linspace(0.,1.,21) ]
        return sorted(fscores, key=lambda x:x[1])[-1][0]



    def predict(self, X, preds=None):
        if preds is None:
            preds = self.model.predict(X)
        return (preds > self.thresholds).astype(int)
