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
            'ultima_marca_comprada': "cat",
            'id_marca': "cat",
            'precio_compra_median': "impute",
            'ratio_promo': "impute",
            'precio_compra_min': "impute",
            'total_dia_sum': "impute",
            'precio_compra_max': "impute",
            'dias_ult_compra_full_max': "impute",
            'total_dia_median': "impute",
            'dias_ult_compra_full_median': "impute",
            'dias_ult_visita_max': "impute",
            'total_dia_min': "impute",
            'dias_ult_compra_median': "impute",
            'dias_ult_compra_max': "impute",
        },
        {
            'ultima_marca_comprada': "cat",
            'id_marca': "cat",
            'total_dia_median': "impute",
            'total_dia_sum': "impute",
            'edad': "impute",
            'dias_ult_compra_max': "impute",
            'dias_ult_compra_full_max': "impute",
            'ingreso_anual': "impute",
            'ratio_promo': "impute",
            'id_marca_mode': "impute",
            'total_dia_min': "impute",
            'dias_ult_visita_max': "impute",
        },
        {
            'id_marca':"cat",
            'precio_compra_max':"impute",
            'total_dia_min':"impute",
            'ultima_marca_comprada':"impute",
            'total_dia_sum':"impute",
            'dias_ult_compra_full_median':"impute",
            'dias_ult_compra_full_max':"impute",
            'ingreso_anual':"impute",
            'dias_ult_compra_median':"impute",
            'total_dia_median':"impute",
            'precio_compra_median':"impute",
            'precio_compra_min':"impute",
            'dias_ult_visita_max':"impute",
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