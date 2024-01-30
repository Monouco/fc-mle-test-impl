# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import joblib
import os

import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from google.cloud import storage
from contextlib import asynccontextmanager

app = FastAPI()
client = storage.Client()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_model_cluster = None
_model_regressors = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    with open("model.joblib", "wb") as f:
        client.download_blob_to_file(f"{os.environ['AIP_STORAGE_URI']}/model_cluster.joblib", f)
        client.download_blob_to_file(f"{os.environ['AIP_STORAGE_URI']}/model_regressor_0.joblib", f)
        client.download_blob_to_file(f"{os.environ['AIP_STORAGE_URI']}/model_regressor_1.joblib", f)
        client.download_blob_to_file(f"{os.environ['AIP_STORAGE_URI']}/model_regressor_2.joblib", f)

    _model_cluster = joblib.load("model_cluster.joblib")
    _model_regressors = [
        joblib.load("model_regressor_0.joblib"),
        joblib.load("model_regressor_1.joblib"),
        joblib.load("model_regressor_2.joblib"),
    ]
    yield

@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/healthz"))
def health():
    return {}

@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/readiness"))
def readiness():
    return {}


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
async def predict(request: Request):
    body = await request.json()

    instances = body["instances"]
    inputs_df = pd.DataFrame(instances)
    inputs_df.replace({None: np.nan}, inplace=True)
    preprocessed_cluster_cols = list(_model_cluster["preprocess"].feature_names_in_)

    inputs_df[preprocessed_cluster_cols] = _model_cluster["preprocess"].transform(inputs_df[preprocessed_cluster_cols])

    clusters = _model_cluster.predict(inputs_df[preprocessed_cluster_cols])

    inputs_df["cluster"] = clusters
    preprocessed_regressor_cols = [list(_model_regressor["preprocess"].feature_names_in_) \
                                   for _model_regressor in _model_regressors
                                   ]

    for i in range(0,3):
        dataset = inputs_df[inputs_df["cluster"]==i]
        dataset[preprocessed_regressor_cols] = _model_regressors[i]["preprocess"].transform(dataset[preprocessed_regressor_cols])
        dataset["precio"] = _model_regressors[i]["model"].predict(dataset[preprocessed_regressor_cols])

    return {"predictions": inputs_df}