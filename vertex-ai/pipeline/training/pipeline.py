
import pathlib
import pandas as pd

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output
from config import PipelineConfig
from typing import Tuple
#from utils.upload_model import upload_model


config = PipelineConfig()

@dsl.component(
    base_image="us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/pipeline-training-env:default"
)
def save_to_gcs(
    model: Input[Model],
    dest_bucket:str,
    dest_path: str,
    model_name: str,
) -> str:
    from google.cloud import storage
    storage_client = storage.Client()
    model_path = model.path
    print(model_path)
    origin_bucket = model_path.split('gcs/')[1].split('/')[0]
    origin_path = model_path.split(origin_bucket)[1][1:]
    source_bucket = storage_client.bucket(origin_bucket)
    source_blob = source_bucket.blob(f"{origin_path}/{model_name}.joblib")
    destination_bucket = storage_client.bucket(dest_bucket)
    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, f"{dest_path}/{model_name}.joblib", 
    )
    return f"gs://{dest_bucket}/{dest_path}/{model_name}.joblib"

@dsl.component(
    base_image="us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/pipeline-training-env:default"
)
def enrich_data(
    data_path: str,
    data_compras: Output[Dataset],
    data_cliente: Output[Dataset],
) :
    import pandas as pd
    from utils.utils import gen_client_variables
    df = pd.read_csv(data_path)
    df_data_compras, df_data_cliente = gen_client_variables(df)
    df_data_compras.to_csv(data_compras.path)
    df_data_cliente.to_csv(data_cliente.path)

@dsl.component(
        base_image="us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/pipeline-training-env:default"
)
def join_cluster_compra(
    data_compras: Input[Dataset],
    data_cliente: Input[Dataset],
    data_compras_joined: Output[Dataset],
) :
    import pandas as pd
    product_columns = ['id', 'incidencia_compra', 'id_marca', 'cantidad',
       'ultima_marca_comprada', 'ultima_cantidad_comprada', 'precio_marca_1',
       'precio_marca_2', 'precio_marca_3', 'precio_marca_4', 'precio_marca_5',
       'promo_marca_1', 'promo_marca_2', 'promo_marca_3', 'promo_marca_4',
       'promo_marca_5','precio_compra', 'compra_promo']
    df_compras = pd.read_csv(data_compras.path)
    df_clientes = pd.read_csv(data_cliente.path)
    dataset_scaled = df_clientes.merge(df_compras[product_columns],left_on="id",right_on="id")
    dataset_scaled.to_csv(data_compras_joined.path)

    

@dsl.container_component
def train_cluster(
    data: Input[Dataset],
    model: Output[Model],
    data_preprocessed: Output[Dataset],
    hparams: dict,
):
    return dsl.ContainerSpec(
        image=config.train_cluster_container_uri,
        command=["python", "-m"],
        args=[
            "src.main",
            "--data",
            data.path,
            "--model",
            model.path,
            "--hparams",
            hparams,
            "--data_out",
            data_preprocessed.path
        ],
    )

@dsl.container_component
def train_regressor(
    data: Input[Dataset],
    model_cluster:int,
    model: Output[Model],
    test_data: Output[Dataset],
    metrics: Output[Metrics],
    hparams: dict,
    columns:dict,
):
    return dsl.ContainerSpec(
        image=config.train_regressor_container_uri,
        command=["python", "-m"],
        args=[
            "src.main",
            "--data",
            data.path,
            "--data_test",
            test_data.path,
            "--model",
            model.path,
            "--metric",
            metrics.path,
            "--cluster",
            model_cluster,
            "--hparams",
            hparams,
            "--columns",
            columns
        ],
    )


@dsl.pipeline(name=config.pipeline_name)
def pipeline(
    project_id: str = config.project_id,
    project_location: str = config.project_location,
    dataset_url: str = config.dataset_url
):
    """
    XGB training pipeline which:
     1. Splits and extracts a dataset from BQ to GCS
     2. Trains a model via Vertex AI CustomContainerTrainingJob
     3. Evaluates the model against the current champion model
     4. If better the model becomes the new default model

    Args:
        project_id (str): project id of the Google Cloud project
        project_location (str): location of the Google Cloud project
        model_name (str): name of model
        dataset_id (str): id of BQ dataset used to store all staging data
        dataset_location (str): location of dataset
    """

    # data extraction to gcs
    enrich_op = enrich_data(data_path=dataset_url
                            ).set_display_name("Enrich")

    train_cluster_model = (
        train_cluster(
            data=enrich_op.outputs["data_cliente"],
            hparams=config.cluster_params,
        )
        .after(enrich_op)
        .set_cpu_limit("8")
        .set_display_name("Train Cluster Model")
    )

    join_datasets_op = (
        join_cluster_compra(
            data_compras=enrich_op.outputs["data_compras"],
            data_cliente=train_cluster_model.outputs["data_preprocessed"],
        )
        .after(train_cluster_model)
        .set_display_name("Join clusters with compras")
    )


    for i in range(0,3):
        params = config.linear_models_params[i]
        columns = config.linear_models_columns[i]
        train_model = (
            train_regressor(
                data=join_datasets_op.outputs["data_compras_joined"],
                model_cluster=i,
                hparams=params,
                columns=columns
            ).after(join_datasets_op).set_display_name(f"Regressor Model {i}")
        )
#        _ = upload_model(
#            project_id=project_id,
#            location=project_location,
#            test_data=train_model.outputs["test_data"],
#            model=train_model.outputs["model"],
#            model_evaluation=train_model.outputs["metrics"],
#            eval_metric=config.regressor_primary_metric,
#            eval_lower_is_better = False,
#            comparable=True,
#            model_name=f"model_regressor_{i}",
#            pipeline_job_id="{{$.pipeline_job_name}}",
#        ).set_display_name("Upload model")
        save_to_gcs(
            model=train_model.outputs["model"],
            model_name=f"model_regressor_{i}",
            dest_bucket=config.dest_bucket,
            dest_path=config.dest_artifact_path
        )

#    _ = upload_model(
#        project_id=project_id,
#        location=project_location,
#        test_data=enrich_op.outputs["data_cliente"],
#        model=train_model.outputs["model"],
#        model_name="model_cluster",
#        pipeline_job_id="{{$.pipeline_job_name}}",
#        comparable=False,
#    ).set_display_name("Upload model")
    save_to_gcs(
        model=train_cluster_model.outputs["model"],
        model_name="model_cluster",
        dest_bucket=config.dest_bucket,
        dest_path=config.dest_artifact_path
    )
