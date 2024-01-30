import pandas as pd
from typing import Tuple

def get_main_item(row: pd.Series) -> int:
    try:
        return int(row["id_marca_mode"])
    except:
        return int(row["id_marca_max_count"])

def get_price(row: pd.Series) -> float:
    column_mark = ["precio_marca_1","precio_marca_2",
                   "precio_marca_3","precio_marca_4",
                   "precio_marca_5"]
    product = int(row["id_marca"])
    if product ==0:
        return 0
    return row[column_mark[product-1]] * row["cantidad"]

def bought_at_sale(row: pd.Series) -> int:
    column_mark = ["promo_marca_1","promo_marca_2",
                   "promo_marca_3","promo_marca_4","promo_marca_5"]
    if int(row["cantidad"]) == 0:
        return 0
    for column in column_mark:
        got_promo = int(row[column])
        if got_promo == 1:
            return 1
    return 0

def bought_price(row: pd.Series) -> int:
    column_mark = ["precio_marca_1","precio_marca_2",
                   "precio_marca_3","precio_marca_4","precio_marca_5"]
    product = int(row["id_marca"])
    if product ==0:
        return 0
    return row[column_mark[product-1]]

def days_till_event(data_compras :pd.DataFrame) -> pd.DataFrame:
    data_compras["dias_ult_visita"] = 0
    data_compras["dias_ult_compra"] = 0
    data_compras["dias_ult_promo"] = 0
    data_compras["dias_ult_compra_full"] = 0
    for i in range(0,data_compras.shape[0]):
        cur_row = data_compras.iloc[i]
        if i <1 or cur_row["id"] != data_compras.iloc[i-1]["id"]:
            ult_visita = int(cur_row["dia_visita"])
            ult_compra = int(cur_row["dia_visita"])
            ult_promo = int(cur_row["dia_visita"])
            ult_compra_full = int(cur_row["dia_visita"])
        else:
            prev_row = data_compras.iloc[i-1]
            ult_visita = int(cur_row["dia_visita"]) - int(prev_row["dia_visita"])
            ult_compra = ult_visita + int(prev_row["dias_ult_compra"]) if int(prev_row["incidencia_compra"])==0 else 0
            ult_promo = ult_visita + int(prev_row["dias_ult_promo"]) if int(prev_row["compra_promo"])==0 else 0
            ult_compra_full = ult_visita + int(prev_row["dias_ult_compra_full"]) if int(prev_row["incidencia_compra"])==0 or int(prev_row["compra_promo"]) == 1  else 0
        data_compras.at[i,"dias_ult_visita"] = ult_visita
        data_compras.at[i,"dias_ult_compra"] = ult_compra
        data_compras.at[i,"dias_ult_promo"] = ult_promo
        data_compras.at[i,"dias_ult_compra_full"] = ult_compra_full
    return data_compras

def bought_variables(data_compras:pd.DataFrame, client_columns:list) -> pd.DataFrame:
    data_cliente_compra = data_compras.loc[data_compras["incidencia_compra"]>0]
    data_cliente_compra_producto = data_cliente_compra.groupby(client_columns + ["id_marca"]).agg(
        cantidad_cum = ("cantidad","sum"),
    ).reset_index()
    indices = data_cliente_compra_producto.groupby(client_columns)['cantidad_cum'].idxmax().reset_index(drop=True)
    data_producto_comprado = data_cliente_compra_producto.merge(right=indices, 
                                                                how="inner", 
                                                                left_index=True, 
                                                                right_on="cantidad_cum")[["id","id_marca"]]
    data_producto_comprado.rename(columns={"id_marca":"id_marca_max_count"},
                                  inplace=True)
    return data_producto_comprado

def gen_client_aggregations(data_cliente_compra:pd.DataFrame, client_columns:list) -> pd.DataFrame:
    data_cliente_left = data_cliente_compra.groupby(client_columns).agg(
        total_dia_max = ("total_dia", "max"),
        total_dia_min = ("total_dia", "min"),
        total_dia_mean = ("total_dia", "mean"),
        total_dia_median = ("total_dia", "median"),
        total_dia_sum = ("total_dia", "sum"),
        incidencia_compra_sum = ("incidencia_compra", "sum"),
        compra_promo_sum = ("compra_promo", "sum"),
        id_marca_mode = ("id_marca", pd.Series.mode),
        dias_ult_visita_max = ("dias_ult_visita", "max"),
        dias_ult_visita_min = ("dias_ult_visita", "min"),
        dias_ult_visita_median = ("dias_ult_visita", "median"),
        dias_ult_visita_mean = ("dias_ult_visita", "mean"),
        dias_ult_compra_max = ("dias_ult_compra", "max"),
        dias_ult_compra_median = ("dias_ult_compra", "median"),
        dias_ult_promo_max = ("dias_ult_promo", "max"),
        dias_ult_promo_median = ("dias_ult_promo", "median"),
        dias_ult_compra_full_max = ("dias_ult_compra_full", "max"),
        dias_ult_compra_full_median = ("dias_ult_compra_full", "median"),
        precio_compra_max = ("precio_compra", "max"),
        precio_compra_min = ("precio_compra", "min"),
        precio_compra_median = ("precio_compra", "median"),
    ).reset_index()
    return data_cliente_left

def gen_client_variables(df: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                    pd.DataFrame]:
    client_columns = ["id","genero","estado_civil","edad",
                      "nivel_educacion","ingreso_anual","ocupacion"]
    df["total_dia"] = df.apply(get_price,axis=1)
    df["compra_promo"] = df.apply(bought_at_sale,axis=1)
    df["precio_compra"] = df.apply(bought_price,axis=1)
    df = days_till_event(df)
    data_bought = bought_variables(df,client_columns)
    data_agg = gen_client_aggregations(df,client_columns)
    data_cliente = data_agg.merge(data_bought,left_on="id",right_on="id")
    data_cliente["id_marca_mode"] = data_cliente.apply(get_main_item,axis=1)
    data_cliente["ratio_promo"] = data_cliente["compra_promo_sum"] / data_cliente["incidencia_compra_sum"]
    return df, data_cliente