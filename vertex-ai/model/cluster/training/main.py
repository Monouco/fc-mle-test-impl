import argparse
import os
import logging
import json
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

from .preprocess import transformes
from .config import ModelConfig, PreprocessConfig
from utilities.utils import save_model_artifact, save_training_dataset_metadata
logging.basicConfig(level=logging.INFO)

def create_model(hiper_parameters: dict) ->KMeans:
    return KMeans(**hiper_parameters)

def build_ml_pipeline(model:KMeans, preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model",model)
        ]
    )

def fit_pipeline(pipeline: Pipeline,
                 X_train: pd.DataFrame,
                 preprocess_cols: list,
                 cluster_cols: list) -> None:
    X_train_preprocessed = X_train.copy()
    X_train_preprocessed[preprocess_cols] = pipeline["preprocess"].fit_transform(X_train[preprocess_cols])
    pipeline["model"].fit(X_train_preprocessed[cluster_cols])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="")
    parser.add_argument("--hparams", default={}, type=json.loads)
    parser.add_argument("--data_out", type=str, required=True)

    args = parser.parse_args()
    model_config = ModelConfig()
    preprocess_config = PreprocessConfig()
    logging.info("Read csv files into dataframes")
    X_train = pd.read_csv(args.data)
    
    logging.info("Initialize preprocessor")
    preprocess_flow = transformes.instance_column_transformer(preprocess_config.columns_preprocess_transformers)

    logging.info("Build sklearn pipeline with KMeans model")
    k_means_model = create_model(args.hparams)
    pipeline = build_ml_pipeline(model=k_means_model,
                                 preprocess=preprocess_flow)

    logging.info("Fit ml pipeline")
    fit_pipeline(pipeline, X_train,
                preprocess_cols=preprocess_config.columns_preprocess,
                cluster_cols=model_config.columns_for_model)

    X_train_post = X_train[["id"] + preprocess_config.columns_preprocess].copy()
    X_train_post[preprocess_config.columns_preprocess] = pipeline["preprocess"].transform(X_train)
    print(X_train_post.shape)
    X_train_post["cluster"] = pipeline["model"].predict(X_train_post[model_config.columns_for_model])
    print(X_train_post.shape)
    logging.info(f"Save pipeline")
    save_model_artifact(pipeline, args.model, "model_cluster")

    logging.info(f"Persist URIs of training file(s) for model monitoring in batch predictions")
    save_training_dataset_metadata(args.model, args.data)

    X_train_post.to_csv(args.data_out)

if __name__ == "__main__":
    main()

