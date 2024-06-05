"""
    Module to manage optuna weights for stacking
"""

import os
import sys

from functools import partial

import numpy as np
import optuna

from sklearn.metrics import log_loss

class OptunaWeights:
    """
        Class to optimize or load weights for stacking models
    """
    def __init__(self, random_state, weights=None) -> any:
        self.study = None
        self.weights = weights
        self.random_state = random_state

    def _objective(self, trial, y_true, y_preds):
        # define weights for preds of each model
        weights = [trial.suggest_float(f'weights{n}', 0, 1) for n in range(len(y_preds))]
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
        # calculate logloss for weighted prediction
        score = log_loss(y_true, weighted_pred)

        return score
    
    def fit(self, y_true, y_preds, n_trials=300):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name='optuna-weights', direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f'weights{n}'] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, weights must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)

        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=300):
        self.fit(y_true, y_preds, n_trials=n_trials)
        
        return self.predict(y_preds)

    def weights(self,):
        return self.weights 