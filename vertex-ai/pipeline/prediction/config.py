from os import environ as env
from dataclasses import dataclass
@dataclass
class PipelineConfig:
    pipeline_name = "price_stimator"
    project_id = env.get("VERTEX_PROJECT_ID")
    project_location=env.get("VERTEX_PROJECT_LOCATION")
    dataset_url = "gs://ue4_vertex_bucket/datasets/compras_data.csv"
    train_cluster_container_uri = "us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-cluster-model/cluster-training:default"
    train_regressor_container_uri = "us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-regressor-model/regression-training:default"
    serving_container_uri = ""
    prediction_path = "gs://ue4_vertex_bucket/PREDICTIONS/price_estimator"
    prediction_name="predictions"

    model_path ="gs://ue4_vertex_bucket/MODELS/price_estimator"
    number_clusters = 3