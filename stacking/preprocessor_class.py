"""
    Module for preprocessing dataframes for stacking models
"""

# General libs
from itertools import combinations

# ML
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import OneHotEncoder

class Preprocessor:
    """
        Main class
    """
    def __init__(self, num_cols=None, max_pattern=2, pipe=True) -> any:
        self.num_cols = num_cols
        self.max_pattern = max_pattern
        self.pipe = pipe

    def preprocess(self, X_train, X_test, text_col=None):
        """
            Preprocess x_train and x_test based on sklearn pipeline consisting of:
                - numcols: imputing >> scaling
                - catcols: imputing >> encoding
                - textcols (optional): if provided vectorized using TFIDF
            Params:
                X_train: pd.DataFrame. train data
                X_test: pd.DataFrame. test data
                - text_col (optional): str. text column name
        """
        X_train = self.create_numeric_combinations(X_train.copy(deep=True))
        X_test = self.create_numeric_combinations(X_test.copy(deep=True))

        if self.pipe:
            num_cols = X_train.select_dtypes(include=['int', 'float']).columns
            cat_cols = X_train.select_dtypes(include=['object', 'bool', 'category']).columns
            if text_col in X_train.columns:
                cat_cols = cat_cols.drop(text_col)
            
            tfidf_params = {
                'min_df': 10,
                'max_df': 95
            }
            vect = TfidfVectorizer(**tfidf_params)
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='nan')),
                ('encoder', OneHotEncoder(handle_missing='value', handle_unknown='value'))
            ])

            if text_col in X_train.columns:
                preprocessor = ColumnTransformer([
                    ('numerical', num_pipeline, num_cols),
                    ('categorical', cat_pipeline, cat_cols),
                    ('countvect', vect, text_col)
                ])
            preprocessor = ColumnTransformer([
                    ('numerical', num_pipeline, num_cols),
                    ('categorical', cat_pipeline, cat_cols)
                ])
            X_train = self.pipe.fit_transform(X_train)
            X_test = self.pipe.transform(X_test)
        
        else:
            num_types = ['int', 'float']
            num_cols = [_ for _ in X_train.columns if X_train[_].dtype in num_types]
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        return X_train, X_test
        
    def create_numeric_combinations(self, df):
        """
            Creates new columns based on combinations of numerical columns
        """
        new_cols = []
        for comb in range(2, len(self.num_cols) + 1):
            for col in combinations(self.num_cols, comb):
                if len(col) > self.max_pattern:
                    break
                
                col_names = list(col)
                new_col = '_'.join(col_names) + '_mult'
                df[new_col] = df[col_names[0]]
                for c in col_names[1:]:
                    df[new_col] *= df[c]
                new_cols.append(new_col)
        
        return df
    
    def get_pipe(self,):
        """
            Returns trained pipeline
        """
        return self.pipe