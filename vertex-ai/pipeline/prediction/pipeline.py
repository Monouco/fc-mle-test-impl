
import pathlib
import pandas as pd

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output
from config import PipelineConfig
from typing import Tuple
#from vertex_components import upload_model


config = PipelineConfig()


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
       'promo_marca_5', 'compra_promo']
    df_compras = pd.read_csv(data_compras.path)
    df_clientes = pd.read_csv(data_cliente.path)
    dataset_scaled = df_clientes.merge(df_compras[product_columns],left_on="id",right_on="id")
    dataset_scaled.to_csv(data_compras_joined.path)

@dsl.component(
        base_image="us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/pipeline-training-env:default"
)
def cluster(
    data_cliente: Input[Dataset],
    data_clustered: Output[Dataset],
    model_path: str
) :
    import joblib
    import pandas as pd
    from google.cloud import storage
    client = storage.Client()
    with open("model_cluster.joblib", "wb") as f:
        client.download_blob_to_file(f"{model_path}/model_cluster.joblib", f)
    model = joblib.load("model_cluster.joblib")

    inputs_df = pd.read_csv(data_cliente.path)

    preprocessed_cluster_cols = list(model["preprocess"].feature_names_in_)

    preprocess_df = inputs_df.copy()

    preprocess_df[preprocessed_cluster_cols] = model["preprocess"].transform(preprocess_df[preprocessed_cluster_cols])

    trained_features = list(model["model"].feature_names_in_)
    clusters = model["model"].predict(preprocess_df[trained_features])

    inputs_df[preprocessed_cluster_cols] = preprocess_df[preprocessed_cluster_cols] 
    inputs_df["cluster"] = clusters

    inputs_df.to_csv(data_clustered.path)
    
@dsl.component(
        base_image="us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/pipeline-training-env:default"
)
def regression(
    data: Input[Dataset],
    data_prices: Output[Dataset],
    model_path: str,
    cluster: int,
) :
    import joblib
    import pandas as pd
    from google.cloud import storage
    client = storage.Client()
    with open("model_regressor.joblib", "wb") as f:
        client.download_blob_to_file(f"{model_path}/model_regressor_{cluster}.joblib", f)
    model = joblib.load("model_regressor.joblib")

    products = [1,2,3,4,5]
    inputs_df = pd.read_csv(data.path)

    inputs_df["id_marca"] = [products]* len(inputs_df)
    inputs_df = inputs_df.explode("id_marca")
    preprocessed_cluster_cols = list(model["preprocess"].feature_names_in_)

    prices = model.predict(inputs_df[preprocessed_cluster_cols])

    inputs_df["prices"] = prices

    inputs_df[["id","cluster","prices","id_marca"]].to_csv(data_prices.path)
    
@dsl.component(
        base_image="us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/pipeline-training-env:default"
)
def join_predictions(
    data_cluster_0: Input[Dataset],
    data_cluster_1: Input[Dataset],
    data_cluster_2: Input[Dataset],
    data_joined: Output[Dataset],
) :
    import pandas as pd

    cluster_0 = pd.read_csv(data_cluster_0.path)
    cluster_1 = pd.read_csv(data_cluster_1.path)
    cluster_2 = pd.read_csv(data_cluster_2.path)
    joined = pd.concat([cluster_0,cluster_1,cluster_2])
    joined.to_csv(data_joined.path)

@dsl.component(
        base_image="us-east4-docker.pkg.dev/sanguine-anthem-412615/ue4-price-models/pipeline-training-env:default"
)
def save_predictions(
    data: Input[Dataset],
    path: str,
    file_name: str
) -> str:
    import pandas as pd
    save_path = f"{path}/{file_name}.csv"
    df = pd.read_csv(data.path)
    df.to_csv(save_path)
    return save_path
    


@dsl.pipeline(name=config.pipeline_name)
def pipeline(
    project_id: str = config.project_id,
    project_location: str = config.project_location,
#    model_name: str = config.model_registry_name,
    dataset_url: str = config.dataset_url
):
    """

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

    cluster_predict = (
        cluster(
            data_cliente=enrich_op.outputs["data_cliente"],
            model_path=config.model_path
        )
        .after(enrich_op)
        .set_cpu_limit("8")
        .set_display_name("Run Cluster Model")
    )

    join_datasets_op = (
        join_cluster_compra(
            data_compras=enrich_op.outputs["data_compras"],
            data_cliente=cluster_predict.outputs["data_clustered"],
        )
        .after(cluster_predict)
        .set_display_name("Join clusters with compras")
    )

    model_outputs = []

    for i in range(0,3):
        model_output = (
            regression(
                data=join_datasets_op.outputs["data_compras_joined"],
                cluster=i,
                model_path=config.model_path
            ).after(join_datasets_op).set_display_name(f"Run Regressor Model {i}")
        )
        model_outputs.append(model_output.outputs["data_prices"])
    
    join_predictions_op = join_predictions(data_cluster_0=model_outputs[0],
                                           data_cluster_1=model_outputs[1],
                                           data_cluster_2=model_outputs[2])
    save_predictions(data=join_predictions_op.outputs["data_joined"],
                     path=config.prediction_path,
                     file_name=config.prediction_name)
