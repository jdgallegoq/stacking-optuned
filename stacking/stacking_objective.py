import datetime
import optuna

from sklearn.metrics import log_loss

# preprocessor and splitter class
from splitter_class import Splitter
from preprocessor_class import Preprocessor

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

        base_model_num = self.StackingClassifier(
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
                stacking_clf = self.StackingClassifier(
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
    
    def optimize(self, n_trials=300, return_df=False):
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

        if return_df:
            trials_df = optuna.load_study(
                study_name=study_name,
                storage=storage_name
            ).trials_dataframe()

            return trials_df

    def get_best_params(self,):
        """
            Gets best params for each meta model and base model
        """
        trials_df = self.optimize(return_df=True)

        # get best trial fom study
        best = trials_df.iloc[trials_df.value.idxmin()]

        # get best params per model
        # classification params
        xgb_class_opt_params = {
            "n_estimators": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_n_estimators'],
            "max_depth": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_max_depth'],
            "learning_rate": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_learning_rate'],
            "min_child_weight": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_min_child_weight'],
            "gamma": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_gamma'],
            "subsample": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_subsample'],
            "colsample_bytree": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_colsample_bytree'],
            "reg_alpha": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_reg_alpha'],
            "reg_lambda": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_reg_lambda'],
            "scale_pos_weight": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_class' in idx
            }['params_xgb_class_scale_pos_weight'],
            "n_jobs": -2,
            "objective": "binary:logistic",
            "verbosity": self.verbose,
            "eval_metric": "logloss",
            "random_state": self.random_state
        }
        cb_class_opt_params = {
            "iterations": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_class' in idx
            }['params_cb_class_iterations'],
            "depth": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_class' in idx
            }['params_cb_class_depth'],
            "learning_rate": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_class' in idx
            }['params_cb_class_learning_rate'],
            "l2_leaf_reg": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_class' in idx
            }['params_cb_class_l2_leaf_reg'],
            "random_strength": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_class' in idx
            }['params_cb_class_random_strength'],
            "grow_policy": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_class' in idx
            }['params_cb_class_grow_policy'],
            "boostrap_type": "Bayesian",
            "od_type": "Iter",
            "loss_function": "LogLoss",
            "task_type": self.device,
            "random_state": self.random_state 
        }
        lgb_class_opt_params = {
            "num_trees": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_class' in idx
            }['params_lgb_class_num_trees'],
            "max_depth": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_class' in idx
            }['params_lgb_class_max_depth'],
            "lambda_l1": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_class' in idx
            }['params_lgb_class_lambda_l1'],
            "max_bin": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_class' in idx
            }['params_lgb_class_max_bin'],
            "n_jobs":-2,
            "learning_rate": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_class' in idx
            }['params_lgb_class_learning_rate'],
            "num_leaves": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_class' in idx
            }['params_lgb_class_num_leaves'],
            "is_unbalance": True,
            "data_sample_strategy": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_class' in idx
            }['params_lgb_class_data_sample_strategy'],
            "objective": "binary",
            "metric": "binary_logloss",
            "early_stopping_round": self.early_stopping_rounds,
            "verbose": self.verbose,
            "random_state": self.random_state
        }
        # build this dict to return params
        class_opt_params = {
            "xbg_class_opt": xgb_class_opt_params,
            "cb_class_opt": cb_class_opt_params,
            "lgb_class_opt": lgb_class_opt_params
        }

        # regression params
        xgb_reg_opt_params = {
            "n_estimators": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_n_estimators'],
            "max_depth": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_max_depth'],
            "learning_rate": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_learning_rate'],
            "min_child_weight": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_min_child_weight'],
            "gamma": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_gamma'],
            "subsample": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_subsample'],
            "colsample_bytree": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_colsample_bytree'],
            "reg_alpha": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_reg_alpha'],
            "reg_lambda": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_reg_lambda'],
            "scale_pos_weight": {
                idx:best[f'{idx}'] for idx in best.index if 'xgb_reg' in idx
            }['params_xgb_reg_scale_pos_weight'],
            "n_jobs": -2,
            "objective": "reg:logistic",
            "verbosity": self.verbose,
            "eval_metric": "rmse",
            "random_state": self.random_state
        }
        cb_reg_opt_params = {
            "iterations": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_reg' in idx
            }['params_cb_reg_iterations'],
            "depth": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_reg' in idx
            }['params_cb_reg_depth'],
            "learning_rate": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_reg' in idx
            }['params_cb_reg_learning_rate'],
            "l2_leaf_reg": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_reg' in idx
            }['params_cb_reg_l2_leaf_reg'],
            "random_strength": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_reg' in idx
            }['params_cb_reg_random_strength'],
            "grow_policy": {
                idx:best[f'{idx}'] for idx in best.index if 'cb_reg' in idx
            }['params_cb_reg_grow_policy'],
            "boostrap_type": "Bayesian",
            "od_type": "Iter",
            "loss_function": "RMSE",
            "task_type": self.device,
            "random_state": self.random_state 
        }
        lgb_reg_opt_params = {
            "num_trees": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_reg' in idx
            }['params_lgb_reg_num_trees'],
            "max_depth": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_reg' in idx
            }['params_lgb_reg_max_depth'],
            "lambda_l1": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_reg' in idx
            }['params_lgb_reg_lambda_l1'],
            "max_bin": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_reg' in idx
            }['params_lgb_reg_max_bin'],
            "n_jobs":-2,
            "learning_rate": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_reg' in idx
            }['params_lgb_reg_learning_rate'],
            "num_leaves": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_reg' in idx
            }['params_lgb_reg_num_leaves'],
            "is_unbalance": True,
            "data_sample_strategy": {
                idx:best[f'{idx}'] for idx in best.index if 'lgb_reg' in idx
            }['params_lgb_reg_data_sample_strategy'],
            "objective": "regression",
            "metric": "rmse",
            "early_stopping_round": self.early_stopping_rounds,
            "verbose": self.verbose,
            "random_state": self.random_state
        }
        # build this dict to return params
        reg_opt_params = {
            "xbg_reg_opt": xgb_reg_opt_params,
            "cb_reg_opt": cb_reg_opt_params,
            "lgb_reg_opt": lgb_reg_opt_params
        }

        return class_opt_params, reg_opt_params