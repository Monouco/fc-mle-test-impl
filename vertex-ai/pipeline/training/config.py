from os import environ as env
from dataclasses import dataclass
@dataclass
class PipelineConfig:
    pipeline_name = "price_estimator"
    project_id = env.get("VERTEX_PROJECT_ID")
    project_location=env.get("VERTEX_LOCATION")
    dataset_url = f"gs://{env.get('VERTEX_BUCKET')}/datasets/compras_data.csv"
    train_cluster_container_uri = "us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/cluster-training:default"
    train_regressor_container_uri = "us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/regression-training:default"
    dest_bucket = "ue4_vertex_bucket"
    dest_artifact_path = "MODELS/price_estimator"

    regressor_primary_metric = "mse"

    cluster_params = {
        "n_clusters": 3,
        "random_state":42,
    }

    linear_models_columns = [
        {
            'cantidad': "impute",
            'id_marca': "cat",
            'ultima_marca_comprada': "cat",
            'precio_compra_median': "impute",
            'compra_promo': "cat",
            'precio_compra_min': "impute",
            'total_dia_median': "impute",
            'total_dia_sum': "impute",
            'total_dia_min': "impute",
            'precio_compra_max': "impute",
            'dias_ult_compra_full_max': "impute",
            'dias_ult_promo_median': "impute",
            'ratio_promo': "impute",
            'dias_ult_visita_median': "impute",
            'dias_ult_compra_max': "impute",
        },
        {
            'cantidad': "impute",
            'ultima_marca_comprada': "cat",
            'id_marca': "cat",
            'compra_promo': "cat",
            'total_dia_sum': "impute",
            'ingreso_anual': "impute",
            'edad': "impute",
            'dias_ult_compra_full_median': "impute",
        },
        {
            'cantidad': "impute",
            'id_marca': "cat",
            'ultima_marca_comprada': "cat",
            'compra_promo': "cat",
            'precio_compra_median': "impute",
            'total_dia_min': "impute",
            'total_dia_median': "impute",
            'total_dia_sum': "impute",
            'precio_compra_max': "impute",
            'dias_ult_compra_full_max': "impute",
            'ratio_promo': "impute",
            'dias_ult_compra_max': "impute",
            'precio_compra_min': "impute",
            'dias_ult_promo_median': "impute",
        },
    ]

    linear_models_params = [
        {
            "random_state": 1945,
            "max_depth": 5,
            "num_leaves": 32,
        },
        {
            "random_state": 1945,
            "max_depth": 5,
            "num_leaves": 32,
        },
        {
            "random_state": 1945,
            "max_depth": 5,
            "num_leaves": 32,
        },
    ]