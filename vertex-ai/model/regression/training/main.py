import argparse
import os
import logging
import json
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re

from .preprocess import transformes
from utilities.utils import save_model_artifact, save_training_dataset_metadata, save_metrics
logging.basicConfig(level=logging.INFO)

def create_model(hiper_parameters: dict) ->LGBMRegressor:
    return LGBMRegressor(**hiper_parameters)

def build_ml_pipeline(model:LGBMRegressor, preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model",model)
        ]
    )

def separate_datasets(X_train:pd.DataFrame,y_train:pd.DataFrame) -> tuple:
    X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2,
                                                     stratify=X_train["id_marca"],random_state=42)
    return X_train,X_test,y_train,y_test

def fit_pipeline(pipeline: Pipeline,
                 X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 preprocess_cols) -> None:
    X_train_preprocessed = X_train.copy()
    X_train_preprocessed[preprocess_cols] = pipeline["preprocess"].fit_transform(X_train[preprocess_cols])
    pipeline["model"].fit(X_train_preprocessed[preprocess_cols],y_train)

def get_preprocess_cols(columns:dict) -> dict:
    result_dict = {}
    for key, value in columns.items():
        if value not in result_dict:
            result_dict[value] = [key]
        else:
            result_dict[value].append(key)
    return result_dict

def evaluate_model(pipeline:Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   columns: list) -> dict:
    X_test[columns] = pipeline["preprocess"].transform(X_test)
    y_pred = pipeline["model"].predict(X_test[columns])
    eval_metrics = {
        "mse": mean_squared_error(y_test,y_pred),
    }
    return eval_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--data_test", type=str, required=True)
    parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="model_path")
    parser.add_argument("--metric", type=str, help="metric_path")
    parser.add_argument("--hparams", default={}, type=json.loads)
    parser.add_argument("--columns",type=json.loads, required=True)
    parser.add_argument("--cluster", type=int, required=True)
    args = parser.parse_args()
    columns = args.columns
    columns_preprocess = get_preprocess_cols(columns)
    columns_list = list(columns.keys())
    logging.info("Read csv files into dataframes")
    X_train = pd.read_csv(args.data)
    X_train = X_train[X_train["cluster"]==args.cluster]
    y_train = X_train["precio_compra"]
#    X_train = X_train[columns_list]
    X_train, X_test, y_train, y_test = separate_datasets(X_train,y_train)
    
    logging.info("Initialize preprocessor")
    preprocess_flow = transformes.instance_column_transformer(columns_preprocess)

    logging.info("Build sklearn pipeline with LGBM model")
    model_regressor = create_model(args.hparams)
    pipeline = build_ml_pipeline(model=model_regressor,
                                 preprocess=preprocess_flow)

    logging.info("Fit ml pipeline")
    fit_pipeline(pipeline, X_train[columns_list], y_train,
                preprocess_cols=columns_list,)

    metrics = evaluate_model(pipeline,X_test,y_test,columns_list)
    logging.info(f"Save pipeline")
    save_model_artifact(pipeline, args.model, f"model_regressor_{args.cluster}")

    logging.info(f"Persist URIs of training file(s) for model monitoring in batch predictions")
    save_training_dataset_metadata(args.model, args.data)

    logging.info(f"Saving metrics")
    save_metrics(metrics,args.metric)


if __name__ == "__main__":
    main()

