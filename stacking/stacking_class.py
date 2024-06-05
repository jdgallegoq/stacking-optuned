"""
    Module to define a stacking classifier
"""
import datetime
import numpy as np

# ML
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import xgboost as xgb
import lightgbm as lgb
from catboost import (
    CatBoostClassifier,
    CatBoostRegressor
)

import optuna

# preprocessor and splitter class
from splitter_class import Splitter
from preprocessor_class import Preprocessor
from optuna_weights_class import OptunaWeights

class StackingClassifier:
    """
        Main class to define a stacking classifier
    """
    def __init__(
            self,
            n_estimators=100,
            device='CPU',
            random_state=0,
            early_stopping_round=10,
            verbose=1,
            xgb_class_params=None,
            cb_class_params=None,
            lgb_class_params=None,
            xgb_reg_params=None,
            cb_reg_params=None,
            lgb_reg_params=None
            ) -> any:
        # params for models
        self.xgb_class_opt_params = xgb_class_params
        self.cb_class_opt_params = cb_class_params
        self.lgb_class_opt_params = lgb_class_params
        self.xgb_reg_opt_params = xgb_reg_params
        self.cb_reg_opt_params = cb_reg_params
        self.lgb_reg_opt_params = lgb_reg_params
        
        # other models params
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.early_stopping_round = early_stopping_round
        self.verbose = verbose

        # define base models and meta models
        self.base_models = self._define_base_models()
        self.meta_models = self._define_class_models()
        self.len_base = len(self.base_models)
        self.len_meta = len(self.meta_models)

        # initialize splitter
        self.splitter = Splitter(random_state=self.random_state)
        self.n_splits = self.splitter.get_n_splits()
        self.random_state_list = self.splitter.get_random_state_list()

    def _define_class_models(self,):
        """
            Function to define main models which are the weak classifiers
        """
        class_models = {
            'xgb_class': xgb.XGBClassifier(**self.xgb_class_opt_params),
            'lgb_class': lgb.LGBMClassifier(**self.lgb_class_opt_params),
            'cat_class': CatBoostClassifier(**self.cb_class_opt_params)
        }

        return class_models
    
    def _define_reg_models(self,):
        """
            Function to define the meta models which are going to get the weights
            for each weak classifier
        """
        reg_models = {
            'xgb_reg': xgb.XGBRegressor(**self.xgb_reg_opt_params),
            'lbg_reg': lgb.LGBMRegressor(**self.lgb_reg_opt_params),
            'cat_reg': CatBoostRegressor(**self.cb_reg_opt_params)
        }

        return reg_models

    def _define_add_model(self,):
        """
            Function to define additional models
        """
        add_models = {
            'hgbc_class': ensemble.HistGradientBoostingClassifier(
                max_iter=500,
                max_depth=4,
                random_state=self.random_state
            ),
            'lr_class': LogisticRegression(
                max_iter=1000,
                n_jobs=-2
            ),
            'rf_class': ensemble.RandomForestClassifier(
                n_estimators=100,
                mex_depth=4,
                random_state=self.random_state,
                n_jobs=-2
            )
        }

        return add_models

    def _define_base_models(self,):
        """
            Function to aggrupate all base models to train
        """
        class_models = self._define_class_models()
        reg_models = self._define_reg_models()
        add_models = self._define_add_model()
        base_models = {
            **class_models,
            **reg_models,
            **add_models
        }

        return base_models
    
    def train(self, X_train, X_test, y_train, y_test):
        """
            Function to train stacking model based on XGB, CatBoost and LGBM
        """
        base_model_num = self.len_base

        # initialize empty list and arrays for storing model objects
        models = []
        best_iterations = []
        sores = []
        oof_predss = np.zeros((X_train.shape[0], base_model_num))
        test_predss =  np.zeros((X_test.shape[0], base_model_num))
        
        # loop over train and test
        splitted_data = self.splitter.split_data_train(self.X_train, self.y_train)
        for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitted_data):
            n = i % self.n_splits
            m = i // self.n_splits

            stacking_clf = StackingClassifier(
                device=self.device,
                random_state=self.random_state,
                early_stopping_round=self.early_stopping_rounds,
                xgb_class_params=self.xgb_class_params,
                cb_class_params=self.cb_class_params,
                lgb_class_params=self.lgb_class_params,
                xgb_reg_params=self.xgb_reg_params,
                cb_reg_params=self.cb_reg_params,
                lgb_reg_params=self.lgb_reg_params
            )
            base_models = stacking_clf.base_models

            # initialize list to store loop oof_preds for each base model
            oof_preds = []
            test_preds = []
            for name, model in base_models.items():
                if name in ["rf_class", "hgbc_class", "lr_class"]:
                        model.fit(X_train_, y_train_)
                else:
                    if "LGBM" in str(model):
                        model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)])
                    else:
                        model.fit(
                            X_train_,
                            y_train_,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose=self.verbose
                            )
                if "class" in name:
                    y_val_pred = model.predict_proba(X_val)[:, 1]
                    test_pred = model.predict_proba(self.X_test)[:, 1]
                else:
                    y_val_pred = model.predict(X_val)
                    test_pred = model.predict(self.X_test)
                
                score = log_loss(y_val, y_val_pred)
                print(f"Base model {name} [FOLD-{n} SEED-{self.random_state_list[m]}] LogLoss score: {score:.5f}")

                oof_preds.append(y_val_pred)
                test_preds.append(test_pred)
            
            # stack oof and test preds horizontally for each base model and store in oof_predss and test_predss
            oof_preds = np.column_stack(oof_preds)
            oof_predss[X_val.index] = oof_preds
            test_preds = np.column_stack(test_preds)
            test_predss += test_preds / (self.n_splits * len(self.random_state_list))

            i += 1

        # train meta models on preds
        meta_test_predss = np.zeros(X_test.shape[0])
        ensemble_score = []
        weights = []

        for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitted_data):
            n = i % self.n_splits
            m = i // self.n_splits

            train_index, val_index = X_train_.index, X_val.index
            # use preds from base models as input features
            X_train_ = oof_predss[train_index]
            X_val = oof_predss[val_index]
            # get a set of base models and meta models
            stacking_clf = StackingClassifier(
                device=self.device,
                random_state=self.random_state,
                early_stopping_round=self.early_stopping_rounds,
                xgb_class_params=self.xgb_class_params,
                cb_class_params=self.cb_class_params,
                lgb_class_params=self.lgb_class_params,
                xgb_reg_params=self.xgb_reg_params,
                cb_reg_params=self.cb_reg_params,
                lgb_reg_params=self.lgb_reg_params
            )
            meta_models = stacking_clf.meta_models
            # initialize lists to store preds
            oof_preds = []
            test_preds = []
            # loop over meta models
            for name, model in meta_models.items():
                if name in ["rf_class", "hgbc_class", "lr_class"]:
                        model.fit(X_train_, y_train_)
                else:
                    if "LGBM" in str(model):
                        model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)])
                    else:
                        model.fit(
                            X_train_,
                            y_train_,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose=self.verbose
                            )
                y_val_pred = model.predict_proba(X_val)[:, 1]
                test_pred = model.predict_proba(test_predss)[:, 1]
                score = log_loss(y_val, y_val_pred)
                print(f"Meta model {name} [FOLD-{n} SEED-{self.random_state_list[m]}] LogLoss score: {score:.5f}")

                oof_preds.append(y_val_pred)
                test_pred.append(test_pred)

            # Use optuna to optimize weights of each model
            optweights = OptunaWeights(random_state=self.random_state)
            y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
            score = log_loss(y_val, y_val_pred)
            print(f"Ensemble model [FOLD-{n} SEED-{self.random_state_list[m]}] LogLoss score: {score:.5f}")

            ensemble_score.append(score)
            weights.append(optweights.weights)
            meta_test_preds += optweights.predict(test_preds) / (self.n_splits * len(self.random_state_list))

            i += 1

        return base_models, meta_models


class StackingObjective(object):
    """
        Class for defining optimizing objective for stacking models using Optuna
    """
    def __init__(
            self,
            X_train,
            X_test,
            y_train,
            y_test,
            random_state,
            verbose=0,
            text_col=None,
            n_splits=10,
            early_stopping_rounds=100,
            device='cpu'
            ) -> any:

        self.X_train = X_train
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test

        # initialize splitter instance
        self.splitter = Splitter()

        # define preprocessor
        num_types = ['int', 'float']
        numeric_columns = [_ for _ in self.X_train.dtype in num_types]
        pp = Preprocessor(num_cols=numeric_columns)
        self.X_train, self.X_test = pp.preprocess(
            X_train=self.X_train,
            X_test=self.X_test
        )

        self.text_col = text_col
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device
        self.n_splits = n_splits

    def __call__(self, trial):
        # class params
        xgb_class_params = {
            "n_estimators": trial.suggest_int("xgb_class_n_estimators", 20, 1200),
            "max_depth": trial.suggest_int("xgb_class_max_depth", 2, 64),
            "learning_rate": trial.suggest_float("xgb_class_learning_rate", 0.005, 0.3, step=0.001),
            "min_child_weight": trial.suggest_int("xgb_class_min_child_weight", 1, 100),
            "gamma": trial.suggest_float("xgb_class_gamma", 0, 20),
            "subsample": trial.suggest_float("xgb_class_subsample", 0.2, 1, step=0.01),
            "colsample_bytree": trial.suggest_float("xgb_class_colsample_bytree", 0.3, 1, step=0.01),
            "reg_alpha": trial.suggest_float("xgb_class_reg_alpha", 0, 6, step=0.01),
            "reg_lambda": trial.suggest_float("xgb_class_reg_lambda", 0, 6, step=0.01),
            "scale_pos_weight": trial.suggest_float("xgb_class_scale_pos_weight", 1, 1.2, step=0.01),
            "n_jobs": -2,
            "objective": "binary:logistic",
            "verbosity": self.verbose,
            "eval_metric": "logloss",
            "random_state": self.random_state
        }
        cb_class_params = {
            "iterations": trial.suggest_int("cb_class_iterations", 100, 10000),
            "depth": trial.suggest_int("cb_class_depth", 2, 16),
            "learning_rate": trial.suggest_float("cb_class_learning_rate", 0.005, 0.3, step=0.001),
            "l2_leaf_reg": trial.suggest_float("cb_class_l2_leaf_reg", 1, 6, step=0.5),
            "random_strength": trial.suggest_float("cb_class_random_strength", 1, 6, step=0.5),
            "grow_policy": trial.suggest_categorical("cb_class_grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "boostrap_type": "Bayesian",
            "od_type": "Iter",
            "loss_function": "LogLoss",
            "task_type": self.device,
            "random_state": self.random_state 
        }
        lgb_class_params = {
            "num_trees": trial.suggest_int("lgb_class_num_trees", 100, 10000),
            "max_depth": trial.suggest_int("lgb_class_max_depth", 2, 64),
            "lambda_l1": trial.suggest_float("lgb_class_lambda_l1", 0, 6, step=0.01),
            "max_bin": trial.suggest_int("lgb_class_max_bin", 100, 500),
            "n_jobs":-2,
            "learning_rate": trial.suggest_float("lgb_class_learning_rate", 0.005, 0.3, step=0.001),
            "num_leaves": trial.suggest_int("lgb_class_num_leaves", 10, 100),
            "is_unbalance": True,
            "data_sample_strategy": trial.suggest_categorical("lgb_class_data_sample_strategy", ["bagging", "goss"]),
            "objective": "binary",
            "metric": "binary_logloss",
            "early_stopping_round": self.early_stopping_rounds,
            "verbose": self.verbose,
            "random_state": self.random_state
        }

        # regression params
        xgb_reg_params = {
            "n_estimators": trial.suggest_int("_xgb_reg_n_estimators", 20, 1200),
            "max_depth": trial.suggest_int("_xgb_reg_max_depth", 2, 64),
            "learning_rate": trial.suggest_float("_xgb_reg_learning_rate", 0.005, 0.3, step=0.001),
            "min_child_weight": trial.suggest_int("_xgb_reg_min_child_weight", 1, 100),
            "gamma": trial.suggest_float("_xgb_reg_gamma", 0, 20),
            "subsample": trial.suggest_float("_xgb_reg_subsample", 0.2, 1, step=0.01),
            "colsample_bytree": trial.suggest_float("_xgb_reg_colsample_bytree", 0.3, 1, step=0.01),
            "reg_alpha": trial.suggest_float("_xgb_reg_reg_alpha", 0, 6, step=0.01),
            "reg_lambda": trial.suggest_float("_xgb_reg_reg_lambda", 0, 6, step=0.01),
            "scale_pos_weight": trial.suggest_float("_xgb_reg_scale_pos_weight", 1, 1.2, step=0.01),
            "n_jobs": -2,
            "objective": "reg:logistic",
            "verbosity": self.verbose,
            "eval_metric": "rmse",
            "random_state": self.random_state
        }
        cb_reg_params = {
            "iterations": trial.suggest_int("cb_reg_iterations", 100, 10000),
            "depth": trial.suggest_int("cb_reg_depth", 2, 16),
            "learning_rate": trial.suggest_float("cb_reg_learning_rate", 0.005, 0.3, step=0.001),
            "l2_leaf_reg": trial.suggest_float("cb_reg_l2_leaf_reg", 1, 6, step=0.5),
            "random_strength": trial.suggest_float("cb_reg_random_strength", 1, 6, step=0.5),
            "grow_policy": trial.suggest_categorical("cb_reg_grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "boostrap_type": "Bayesian",
            "od_type": "Iter",
            "loss_function": "RMSE",
            "task_type": self.device,
            "random_state": self.random_state 
        }
        lgb_reg_params = {
            "num_trees": trial.suggest_int("lgb_reg_num_trees", 100, 10000),
            "max_depth": trial.suggest_int("lgb_reg_max_depth", 2, 64),
            "lambda_l1": trial.suggest_float("lgb_reg_lambda_l1", 0, 6, step=0.01),
            "max_bin": trial.suggest_int("lgb_reg_max_bin", 100, 500),
            "n_jobs":-2,
            "learning_rate": trial.suggest_float("lgb_reg_learning_rate", 0.005, 0.3, step=0.001),
            "num_leaves": trial.suggest_int("lgb_reg_num_leaves", 10, 100),
            "is_unbalance": True,
            "data_sample_strategy": trial.suggest_categorical("lgb_reg_data_sample_strategy", ["bagging", "goss"]),
            "objective": "regression",
            "metric": "rmse",
            "early_stopping_round": self.early_stopping_rounds,
            "verbose": self.verbose,
            "random_state": self.random_state
        }

        base_model_num = StackingClassifier(
            device=self.device,
            random_state=self.random_state,
            early_stopping_round=self.early_stopping_rounds,
            xgb_class_params=xgb_class_params,
            cb_class_params=cb_class_params,
            lgb_class_params=lgb_class_params,
            xgb_reg_params=xgb_reg_params,
            cb_reg_params=cb_reg_params,
            lgb_reg_params=lgb_reg_params
        ).len_base
        
        # loop over train and test
        splitted_data = self.splitter.split_data_train(self.X_train, self.y_train)
        for i, (X_train, X_val, y_train, y_val) in enumerate(splitted_data):
                stacking_clf = StackingClassifier(
                    device=self.device,
                    random_state=self.random_state,
                    early_stopping_round=self.early_stopping_rounds,
                    xgb_class_params=xgb_class_params,
                    cb_class_params=cb_class_params,
                    lgb_class_params=lgb_class_params,
                    xgb_reg_params=xgb_reg_params,
                    cb_reg_params=cb_reg_params,
                    lgb_reg_params=lgb_reg_params
                )
                base_models = stacking_clf.base_models

                for name, model in base_models.items():
                    if name in ["rf_class", "hgbc_class", "lr_class"]:
                        model.fit(X_train, y_train)
                    else:
                        if "LGBM" in str(model):
                            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                        else:
                            model.fit(
                                X_train,
                                y_train,
                                eval_set=[(X_val, y_val)],
                                early_stopping_rounds=self.early_stopping_rounds,
                                verbose=self.verbose
                                )
                    if "class" in name:
                        y_val_pred = model.predict_proba(X_val)[:, 1]
                        test_pred = model.predict_proba(self.X_test)[:, 1]
                    else:
                        y_val_pred = model.predict(X_val)
                        test_pred = model.predict(self.X_test)

        return log_loss(y_val, y_val_pred)
    
    def optimize(self, n_trials=300):
        """
            Run Optuna study to get best params
        """
        date = str(datetime.now().date()).replace('-', '')
        study_name = f"stacking_optuna_study{date}"
        storage_name = f"sqlite://{study_name}.db"

        objective = self.__call__()
        sampler = optuna.samplers.TPESampler(multivariate=True)
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=n_trials)
        print("Study done!")

    def get_best_params(self,):
        """
            Gets best params for each meta model and base model
        """
        study = self.optimize()

    #TODO