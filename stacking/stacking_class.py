"""
    Module to define a stacking classifier
"""
import os
import datetime
import joblib
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
from stacking_objective import StackingObjective

class StackingClassifier:
    """
        Main class to define a stacking classifier
    """
    def __init__(
            self,
            X_train,
            X_test,
            y_train,
            y_test,
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
        self.X_train = X_train.copy(deep=True)
        self.X_test = X_test.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.y_test = y_test.copy(deep=True)
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

        # initiliaze preprocessor
        num_types = ['int', 'float']
        num_cols = [_ for _ in self.X_train.columns if self.X_train[_].dtype in num_types]
        self.pp = Preprocessor(num_cols=num_cols)

        # initialize splitter
        self.splitter = Splitter(random_state=self.random_state)
        self.n_splits = self.splitter.get_n_splits()
        self.random_state_list = self.splitter.get_random_state_list()

        # initilize stacking optuna objective
        self.stacking_objective = StackingObjective(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test
        )
    
    def preprocess_data(self, text_col=None):
        """
            Return preprocessed X_train and X_test
        """
        X_train_pp, X_test_pp = self.pp.preprocess(
            self.X_train,
            self.X_test,
            text_col=text_col,
            pipe=True
        )
        return X_train_pp, X_test_pp

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
    
    def get_opt_params(self,):
        """
            Return optimized hyperparams 
        """
        return self.stacking_objective.get_best_params()
    
    def train(self,):
        """
            Function to train stacking model based on XGB, CatBoost and LGBM
        """
        # get best params to train
        class_opt_params, reg_opt_params = self.get_opt_params()

        base_model_num = self.len_base

        # initialize empty list and arrays for storing model objects
        models = []
        best_iterations = []
        sores = []
        oof_predss = np.zeros((self.X_train.shape[0], base_model_num))
        test_predss =  np.zeros((self.X_test.shape[0], base_model_num))
        
        # loop over train and test
        splitted_data = self.splitter.split_data_train(self.X_train, self.y_train)
        for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitted_data):
            n = i % self.n_splits
            m = i // self.n_splits

            stacking_clf = StackingClassifier(
                device=self.device,
                random_state=self.random_state,
                early_stopping_round=self.early_stopping_rounds,
                xgb_class_params=class_opt_params['xbg_class_opt'],
                cb_class_params=class_opt_params['cb_class_opt'],
                lgb_class_params=class_opt_params['lgb_class_opt'],
                xgb_reg_params=reg_opt_params['xbg_reg_opt'],
                cb_reg_params=reg_opt_params['cb_reg_opt'],
                lgb_reg_params=reg_opt_params['lgb_reg_opt']
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
        meta_test_predss = np.zeros(self.X_test.shape[0])
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
                xgb_class_params=class_opt_params['xbg_class_opt'],
                cb_class_params=class_opt_params['cb_class_opt'],
                lgb_class_params=class_opt_params['lgb_class_opt'],
                xgb_reg_params=reg_opt_params['xbg_reg_opt'],
                cb_reg_params=reg_opt_params['cb_reg_opt'],
                lgb_reg_params=reg_opt_params['lgb_reg_opt']
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

        return base_models, meta_models, optweights
    
    def save_models(self, path:str=None):
        """
            Save trained models
        """
        if path is None:
            try:
                os.mkdir("./stacking_opt")
                path = "./stacking_opt"
            except:
                pass
        # get base models and meta models and weights
        base_models, meta_models, optweights = self.train()
        # save preprocessor
        joblib.dump(self.pp.get_pipe(), path+"/preprocessor.joblib")
        # save base models
        for model_name, model in base_models.items():
            joblib.dump(model, path+f'/{model_name}.joblib')
        # save meta models
        for model_name, model in meta_models.items():
            joblib.dump(model, path+f'/{model_name}.joblib')
        # save weights
        joblib.dump(optweights, path+'/optweights.joblib')

    def load_models(self, path:str=None):
        """
            Load trained models
        """
        pass