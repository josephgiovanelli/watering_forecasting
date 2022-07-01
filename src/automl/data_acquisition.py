# OpenML provides several benchmark datasets
import openml
import os

import pandas as pd
import numpy as np


def load_dataset_from_openml(id, decode=False):
    """Load a dataset from openml

    Args:
        id (_type_): we can specify either the id or the name
        decode (bool, optional): openml provides categorical data already encoded (w/ an OrdinalEncoder).
        To simulate a real-case scenario (and have data decoded) turn this flag to True. Defaults to False.

    Returns:
        _type_: X is the data matrix, y is the data label, categorical_indicator is an array w/ a boolean for each feature (True if categorical, False otherwise)
    """
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    if decode:
        cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
        Xt = pd.DataFrame(X)
        Xt[cat_features] = Xt[cat_features].fillna(-1)
        Xt[cat_features] = Xt[cat_features].astype("str")
        Xt[cat_features] = Xt[cat_features].replace("-1", np.nan)
        X = Xt.to_numpy()
    return X, y, categorical_indicator


def load_agro_data_from_csv(path=os.path.join("/", "home", "data")):
    def _ground_potential_reconstruction(df):
        """Ground potential data likely contains missing values.
        Since this kind of data has to be treated properly, we define a custom imputation.
        Data from sensors are sampled every 15 minutes, but we are interested in hourly sampling.
        We are lucky if we have the data at the :00 of the hour at hand , bc we do not have to do nothing.
        Otherwise, we take the value of the previous quarter (15 minutes before).
        This is done until the minute :00 of the previous hour.

        Args:
            df (_type_): input dataframe

        Returns:
            _type_: output dataframe
        """
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
