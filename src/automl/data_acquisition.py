# OpenML provides several benchmark datasets
import openml
import pandas as pd
import numpy as np


def load_dataset_from_openml(id, encode=False):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    if encode:
        cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
        Xt = pd.DataFrame(X)
        Xt[cat_features] = Xt[cat_features].fillna(-1)
        Xt[cat_features] = Xt[cat_features].astype("str")
        Xt[cat_features] = Xt[cat_features].replace("-1", np.nan)
        X = Xt.to_numpy()
    return X, y, categorical_indicator
