from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple

def categorical_transformer() -> Pipeline:
    base_categorical = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy='most_frequent'))
        ]
    )
    return base_categorical

def numerical_transformer() -> Tuple[Pipeline,
                                     Pipeline,
                                     Pipeline,
                                     Pipeline]:
    base_numerical_imputer = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy='median'))
        ]
    )
    base_numerical_standard = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy='median')),
            ("standard", StandardScaler()),
        ]
    )
    base_numerical_normalize = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy='median')),
            ("normalize", MinMaxScaler()),
        ]
    )
    base_numerical_robust = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy='median')),
            ("robust", RobustScaler()),
        ]
    )
    return base_numerical_imputer, base_numerical_standard, \
          base_numerical_normalize, base_numerical_robust

def instance_column_transformer(columns:dict) -> ColumnTransformer:
    p_categorical = categorical_transformer()
    p_num_imp, p_num_standard, p_num_norm, p_num_rob = numerical_transformer()
    preprocess = ColumnTransformer(
        transformers=[
            ("impute",p_num_imp,columns["impute"]),
            ("standard_num",p_num_standard,columns["standard"]),
            ("normalize_num",p_num_norm,columns["normalize"]),
            ("robust_num",p_num_rob,columns["robust"]),
#            ("cat",p_categorical,columns["cat"]),
        ]
    )
    return preprocess