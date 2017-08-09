from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys


class BaseModelFn():
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    # A default implementation of _create_features is just to return features.
    # In case any other transformations are needed, just return a dict of keys->tensors at the end
    def _create_features(self, features):
        return features


    # _create_predictions_dict is the place to implement the inference model and generate a dict of keys->tensors
    # of your predictions.
    @abstractmethod
    def _create_predictions(self, features):
        # probs = ...
        # return { 'probs': probs }
        pass


    # _create_loss is the place to define your loss based on the predictions dict from _create_predictions and
    # the labels that were given to model_fn. This should return a single tensor. This must be implemented by the subclass.
    @abstractmethod
    def _create_loss(self, predictions, labels):
        # loss = ...
        # return loss
        pass


    # _create_optimizer should only return the optimizer instance ot use for minimizing the loss
    # Must be defined by the subclass
    @abstractmethod
    def _create_optimizer(self):
        # return tf.train.XXXOptimizer()
        pass


    # default implementation of summaries - just does nothing.
    def _create_summaries(self):
        pass


    # _create_metrics should return a dict of keys to metrics. default implementation just returns an empty dict.
    def _create_metrics(self, predictions, labels):
        return {}




    def enforce_mandatory_features(self, mandatory, features):
        missing = [ x for x in mandatory if x not in features ]
        if missing:
            raise ValueError(",".join(missing) + " are mandatory features")


    # Usage example:
    # model_fn = MyModelFn()   #  a sub class of BaseModelFn
    # est = tf.contrib.learn.Estimator(model_fn=model_fn(), ...)
    def __call__(self):
        def _model_fn(features, labels, mode, params={}):
            self.params = params
            self.global_step = tf.contrib.framework.get_global_step()
            self.loss = None
            self.train_op = None
            self.metric_ops = None

            with tf.variable_scope('features'):
                self.features = self._create_features(features)

            with tf.variable_scope('model'):
                self.predictions = self._create_predictions(self.features)

            with tf.variable_scope('loss'):
                if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
                    self.loss = self._create_loss(self.predictions, labels)
                    tf.summary.scalar('loss', self.loss)

                if mode == ModeKeys.TRAIN:
                    opt = self._create_optimizer()
                    self.train_op = opt.minimize(self.loss, global_step=self.global_step)

            self._create_summaries()
            self.summaries_op = tf.summary.merge_all()

            if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
                self.metric_ops = self._create_metrics(self.predictions, labels)

            return tf.contrib.learn.ModelFnOps(
                mode=mode,
                predictions=self.predictions,
                loss=self.loss,
                train_op=self.train_op,
                eval_metric_ops=self.metric_ops,
            )

        return _model_fn

