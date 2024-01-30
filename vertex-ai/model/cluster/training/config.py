from dataclasses import dataclass
@dataclass()
class ModelConfig:
    params = {
        "n_clusters": 3,
        "random_state":42,
    }
    columns_for_model = [
        'edad',
        'ingreso_anual',
        'total_dia_min',
        'total_dia_median',
        'total_dia_sum',
        'dias_ult_visita_max',
        'dias_ult_visita_median',
        'dias_ult_compra_max',
        'dias_ult_compra_median',
        'dias_ult_promo_median',
        'dias_ult_compra_full_max',
        'dias_ult_compra_full_median',
        'precio_compra_max',
        'precio_compra_min',
        'precio_compra_median',
        'ratio_promo'
    ]

@dataclass()
class PreprocessConfig:
    columns_preprocess_transformers = {
#        "cat": ["genero","estado_civil","nivel_educacion",
#                "ocupacion","id_marca_mode"],
        "impute": ["ratio_promo",
                    'precio_compra_max',
                    'precio_compra_min',
                    'precio_compra_median',],
        "standard": ["dias_ult_visita_median"],
        "normalize": ["edad"],
        "robust": [
            'ingreso_anual',
            'total_dia_min',
            'total_dia_median',
            'total_dia_sum',
            'dias_ult_visita_max',
            'dias_ult_compra_max',
            'dias_ult_compra_median',
            'dias_ult_promo_median',
            'dias_ult_compra_full_max',
            'dias_ult_compra_full_median'],
    }
    columns_preprocess = [
#        "genero","estado_civil","nivel_educacion",
#        "ocupacion",
#        "id_marca_mode",
        "ratio_promo",
        'precio_compra_max',
        'precio_compra_min',
        'precio_compra_median',
        "dias_ult_visita_median",
        "edad",
        'ingreso_anual',
        'total_dia_min',
        'total_dia_median',
        'total_dia_sum',
        'dias_ult_visita_max',
        'dias_ult_compra_max',
        'dias_ult_compra_median',
        'dias_ult_promo_median',
        'dias_ult_compra_full_max',
        'dias_ult_compra_full_median',
    ]