import pandas as pd
import joblib
import logging
import json
from pathlib import Path
from sklearn.pipeline import Pipeline

def save_model_artifact(model_artifact: Pipeline, model_path: str,
                        model_name:str = "model"):
    """Save the trained model to a file.

    Args:
        model_artifact (Pipeline): The trained model object.
        model_path (str): The path to save the model.

    Returns:
        None
    """
    if model_path.startswith("gs://"):
        model_path = Path("/gcs/" + model_path[5:])
    else:
        model_path = Path(model_path)

    model_path.mkdir(parents=True, exist_ok=True)

    model_path = model_path / f"{model_name}.joblib"

    logging.info(f"Save model to: {model_path}")
    joblib.dump(model_artifact, str(model_path))


def save_metrics(metrics: dict, metrics_path: str):
    """Save the evaluation metrics to a JSON file.

    Args:
        metrics (Dict[str, float]): A dictionary containing the evaluation metrics.
        metrics_path (str): The path to save the metrics.

    Returns:
        None
    """
    with open(metrics_path, "w") as fp:
        logging.info(f"Save metrics to: {metrics_path}")
        json.dump(metrics, fp)


def save_training_dataset_metadata(model_path: str, train_data_path: str):
    """Save training dataset information for model monitoring.

    This function persists URIs of training file(s) for model monitoring in batch predictions.

    Args:
        model_path (str): Path to the model directory.
        train_data_path (str): Path to the training data file.

    Returns:
        None
    """

    training_dataset_info = "training_dataset_info.json"

    if model_path.startswith("gs://"):
        model_path = Path("/gcs/" + model_path[5:])
    else:
        model_path = Path(model_path)

    path = Path(model_path) / training_dataset_info
    training_dataset_for_monitoring = {
        "gcsSource": {"uris": [train_data_path]},
        "dataFormat": "csv",
    }

    logging.info(f"Training dataset info: {training_dataset_for_monitoring}")

    with open(path, "w") as fp:
        logging.info(f"Save training dataset info for model monitoring in: {path}")
        json.dump(training_dataset_for_monitoring, fp)

def save_metrics(metrics: dict, metrics_path: str):
    """Save the evaluation metrics to a JSON file.

    Args:
        metrics (Dict[str, float]): A dictionary containing the evaluation metrics.
        metrics_path (str): The path to save the metrics.

    Returns:
        None
    """
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp)
