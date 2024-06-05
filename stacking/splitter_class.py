"""
    Module to Kfold stacking
"""

import random
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

class Splitter:
    """
        Main class for splitting data to train a stacking classifier
    """
    def __init__(
            self,
            test_size=0.2,
            kfold=True,
            n_splits=5,
            random_state=None,
            random_state_list_len=3
            ) -> any:
        self.test_size = test_size
        self.kfold = kfold
        self.n_splits = n_splits
        self.random_state = random_state

        if random_state:
            random.seed(random_state)
            self.random_state_list = [
               random.randint(0, 1e4) for i in range(random_state_list_len)
            ]
        else:
            self.random_state_list = [
               random.randint(0, 1e4) for i in range(random_state_list_len)
            ]
    
    def split_data_train(self, X, y):
        """
            Function for stratified kfold.
            Params:
                - X: any. predictors
                - y: any. target variable
        """
        if self.kfold:
            for random_state in self.random_state_list:
                kf = StratifiedKFold(
                    n_splits=self.n_splits,
                    random_state=random_state,
                    shuffle=True
                )
                for train_index, val_index in kf.split(X, y):
                    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    else:
                        X_train, X_val = pd.DataFrame(X).iloc[train_index], pd.DataFrame(X).iloc[val_index]
                        y_train, y_val = pd.Series(y).iloc[train_index], pd.Series(y).iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            for random_state in self.random_state_list:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=random_state
                )
                yield X_train, X_val, y_train, y_val
    
    def split_data_deploy(self, X, y=None):
        """
            Function for stratified kfold on deployment.
            Params:
                - X: any. predictors
                - y: any. targets
        """
        if self.kfold:
            if y is None:
                for random_state in self.random_state_list:
                    kf = StratifiedKFold(
                        n_splits=self.n_splits,
                        random_state=random_state,
                        shuffle=True
                    )
                    for train_index, val_index in kf.split(X):
                        if isinstance(X, pd.DataFrame):
                            X_one, X_two = X.iloc[train_index], X.iloc[val_index]
                        else:
                            X_one, X_two = pd.DataFrame(X).iloc[train_index], pd.DataFrame(X).iloc[val_index]
                        yield X_one, X_two
            else:
                for random_state in self.random_state_list:
                    kf = StratifiedKFold(
                        n_splits=self.n_splits,
                        random_state=random_state,
                        shuffle=True
                    )
                    for train_index, val_index in kf.split(X, y):
                        if isinstance(X, pd.DataFrame):
                            X_one, X_two = X.iloc[train_index], X.iloc[val_index]
                            y_one, y_two = y.iloc[train_index], y.iloc[val_index]
                        else:
                            X_one, X_two = pd.DataFrame(X).iloc[train_index], pd.DataFrame(X).iloc[val_index]
                            y_one, y_two = pd.Series(y).iloc[train_index], pd.Series(y).iloc[val_index]
                        yield X_one, X_two, y_one, y_two
        else:
            for random_state in self.random_state_list:
                X_one, X_two, y_one, y_two = train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=random_state
                )
                yield X_one, X_two, y_one, y_two
    
    def get_random_state_list(self,):
        return self.random_state_list
    
    def get_n_splits(self,):
        return self.n_splits