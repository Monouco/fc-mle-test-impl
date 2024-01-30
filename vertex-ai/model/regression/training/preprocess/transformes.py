from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def categorical_transformer() -> Pipeline:
    base_categorical = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy='most_frequent'))
        ]
    )
    return base_categorical

def numerical_transformer() -> Pipeline:
    base_numerical_imputer = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy='median'))
        ]
    )
    return base_numerical_imputer

def instance_column_transformer(columns:dict) -> ColumnTransformer:
    p_categorical = categorical_transformer()
    p_num_imp = numerical_transformer()
    preprocess = ColumnTransformer(
        transformers=[
            ("impute",p_num_imp,columns["impute"]),
            ("cat",p_categorical,columns["cat"]),
        ]
    )
    return preprocess