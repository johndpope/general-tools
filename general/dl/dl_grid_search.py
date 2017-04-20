import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from collections import namedtuple
import general

import logging
log = logging.getLogger(__name__)




Result = namedtuple('Result', ['id', 'params', 'history'])

class DLGridSearch:

    def __init__(self, build_fn, build_params={}, generator_fn=None, generator_params={}):
        if not hasattr(build_fn, '__call__'):
            raise Exception("build_fn should be a function (or callable), not the model or a Pipeline object")
        self.build_fn = build_fn
        self.build_params = build_params
        self.generator_fn = generator_fn
        self.generator_params = generator_params
        self.results = []
        self.report_ = None

    def _default_fit_args(self, user_given_args):
        args = { 'verbose': 2 }
        args.update(user_given_args)
        return args

    def _base_fit(self, fit_func):
        build_pg = ParameterGrid(self.build_params) if self.build_params else [{}]
        generator_pg = ParameterGrid(self.generator_params) if self.generator_params else [{}]

        num_build_options, num_gen_options = len(build_pg), len(generator_pg)
        num_options = num_build_options * num_gen_options
        log.info("got {} generator params options and {} build params options. overall - {} options to try".format(
            num_gen_options, num_build_options, num_options))

        idx = 0
        for current_generator_params in generator_pg:
            for current_build_params in build_pg:
                current_params = current_generator_params.copy()
                current_params.update(current_build_params)
                log.info("fitting model ({}/{}) with params: {}".format(idx+1, num_options, current_params))

                est = general.wrappers.KerasClassifier(self.build_fn, self.generator_fn, **current_params)
                history = fit_func(est)
                self.results.append(Result(idx, current_params, history))
                idx += 1


    def fit(self, X, y, X_val=None, y_val=None, **fitargs):
        args = self._default_fit_args(fitargs)

        if (X_val is None) != (y_val is None):
            raise Exception("provide either both X_val and y_val or none")
        val_data = (X_val, y_val) if X_val is not None else None

        def fit_func(est):
            history = est.fit(X, y, validation_data=val_data, **args)
            return history

        self._base_fit(fit_func)
