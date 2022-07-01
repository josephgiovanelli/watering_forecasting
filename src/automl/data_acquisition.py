# OpenML provides several benchmark datasets
import openml
import os

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


def load_agro_data_from_csv(path=os.path.join("/", "home", "data")):
    def _ground_potential_reconstruction(df):
        dfs = {}
        for minute in [0, 15, 30, 45]:
            dfs[minute] = df[
                df.apply(
                    lambda x: pd.to_datetime(x["timestamp"], unit="s").minute == minute,
                    axis=1,
                )
            ]
            if minute != 0:
                dfs[minute]["timestamp"] = (
                    (dfs[minute]["timestamp"] + 3600) / 3600
                ).astype(int) * 3600
            dfs[minute] = dfs[minute].pivot(
                index="timestamp", columns=["z", "y", "x"], values="value"
            )
        for minute in [45, 30, 15]:
            dfs[0] = dfs[0].combine_first(dfs[minute])
        return dfs[0]

    data = {}
    for variable in [
        "air_humidity",
        "air_temperature",
        "irrigation",
        "precipitation",
        "solar_radiation",
        "wind_speed",
        "ground_potential",
    ]:
        data[variable] = pd.read_csv(os.path.join(path, f"{variable}.csv"))
        if variable == "ground_potential":
            data[variable] = _ground_potential_reconstruction(data[variable].copy())
            data[variable].columns = [
                f"z{-column[0]}_y{column[1]}_x{column[2]}"
                for column in data[variable].columns.values
            ]
        else:
            data[variable] = data[variable].rename(columns={"value": variable})
            data[variable] = data[variable].set_index("timestamp")

    result = pd.concat(list(data.values()), axis=1)
    return result.fillna(method="ffill")
