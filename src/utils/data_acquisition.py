import os

import pandas as pd
import numpy as np

import math

from datetime import datetime, timezone, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf


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


def compute_statistics(window_size):
    """Compute min and max values for each variable, based on the values find on the dataset.

    Args:
        window_size (_type_): the specified window size

    Returns:
        _type_: a dictionary containing dataset statistics
    """
    statistics = {}

    # air temperature, air humidity, wind speed, solar radiation, precipitation, irrigation
    min_w_values = [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_w_values = [40.0, 100.0, 15.0, 1000.0, 20.0, 6.0]

    # ground_potential
    min_gp_value = [-1500.0]
    max_gp_value = [-15.0]

    statistics["min"] = tf.convert_to_tensor(
        np.concatenate(
            (
                np.repeat(min_w_values, window_size),
                np.repeat(min_gp_value, window_size * 12),
            )
        ),
        dtype=tf.float32,
    )
    statistics["max"] = tf.convert_to_tensor(
        np.concatenate(
            (
                np.repeat(max_w_values, window_size),
                np.repeat(max_gp_value, window_size * 12),
            )
        ),
        dtype=tf.float32,
    )

    return statistics


def create_rolling_window(df, window_size, stride=None):
    """Create the rolling window, according to the specified window size and (optional) stride.

    Args:
        df (_type_): input dataset
        window_size (_type_): the specified window size
        stride (_type_, optional): the spcified window stride. Defaults to None.

    Returns:
        _type_: rolling window
    """
    if window_size < 2 or (stride != None and stride < 2):
        print("The window size and the stride must be greater than 1!")
    else:
        m_var_names = list(
            df.columns[:4]
        )  # air_temperature, air_humidity, wind_speed, solar_radiation
        w_var_names = list(df.columns[4:6])  # precipitation, irrigation
        gp_var_names = list(df.columns[6:])  # ground potential

        # Convert row indexes from Unix time to timestamp
        df["timestamp"] = [
            datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=2)))
            for ts in list(df.index)
        ]
        df.set_index("timestamp", drop=True, inplace=True)

        if stride != None:
            # Compute aggregations
            m_var_agg = df[m_var_names].resample(f"{stride}H").mean()
            w_var_agg = df[w_var_names].resample(f"{stride}H").sum()
            gp_var_agg = df[gp_var_names].resample(f"{stride}H").last()

            df = pd.concat([m_var_agg, w_var_agg, gp_var_agg], axis=1)

            # Adjust the window size according to the stride
            window_size = math.ceil(window_size / stride)

        # Create rolling window template
        for step in np.arange(1, window_size):
            for idx, var in enumerate(m_var_names + w_var_names, start=1):
                df.insert(
                    loc=int(step) * idx + (idx - 1),
                    column=f"{var}_{step}",
                    value=np.nan,
                )
            for var in gp_var_names:
                df[f"{var}_{step}"] = np.nan

        # Row indexes reset for the following computations
        timestamps = list(df.index)
        df.reset_index(drop=True, inplace=True)

        # Populate rolling window
        try:
            for idx, _ in df.iterrows():
                for var in m_var_names + w_var_names:
                    df.loc[idx, f"{var}_1":f"{var}_{window_size - 1}"] = df.loc[
                        idx + 1 : idx + window_size - 1, var
                    ].values
                for step in np.arange(1, window_size):
                    df.loc[
                        idx, f"{gp_var_names[0]}_{step}":f"{gp_var_names[-1]}_{step}"
                    ] = df.loc[
                        idx + step, f"{gp_var_names[0]}":f"{gp_var_names[-1]}"
                    ].values
        except (KeyError, ValueError):
            print()

        # Restore previous row indexes
        df["timestamp"] = timestamps
        df.set_index("timestamp", inplace=True)

        # Remove rows with NaN values
        df.dropna(inplace=True)

        return df


def train_val_test_split(
    data, output_horizon, test_ratio, val_ratio=None, shuffle=False
):
    """Split input dataframe in train, validation and test set.

    Args:
        data (_type_): input data to split
        output_horizon (_type_): output horizon (in hours) to predict
        test_ratio (_type_): percentage of test samples
        val_ratio (_type_, optional): percentage of validation samples, taken from train set. Defaults to None.
        shuffle (bool, optional): whether to shuffle samples. Defaults to False.

    Returns:
        _type_: X_train is the training set, y_train is the training label set,
                X_val is the (optional) validation set, y_val is the (optional) validation label set,
                X_test is the test set, y_test is the test label set
    """
    X = data.iloc[:, : -output_horizon * 12].to_numpy()
    y = data.iloc[:, -output_horizon * 12 :].to_numpy()

    X_val = []
    y_val = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, shuffle=shuffle
    )

    if val_ratio != None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=42, shuffle=shuffle
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X_train, y_train, X_val, y_val, window_size):
    """Normalization function used for Scikit-learn regressors.
    MinMaxScaler is used.

    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        y_val (_type_): the validation label set
        window_size (_type_): the specified window size

    Returns:
        _type_: the normalized X_train, y_train, X_val sets and the y_val scaler
    """
    X_train_mv_scaler = MinMaxScaler().fit(X_train[:, : window_size * 6])
    X_train_mv = X_train_mv_scaler.transform(X_train[:, : window_size * 6])

    X_train_gpv_scaler = MinMaxScaler().fit(X_train[:, window_size * 6 :])
    X_train_gpv = X_train_gpv_scaler.transform(X_train[:, window_size * 6 :])

    X_val_mv_scaler = MinMaxScaler().fit(X_val[:, : window_size * 6])
    X_val_mv = X_val_mv_scaler.transform(X_val[:, : window_size * 6])

    X_val_gpv_scaler = MinMaxScaler().fit(X_val[:, window_size * 6 :])
    X_val_gpv = X_val_gpv_scaler.transform(X_val[:, window_size * 6 :])

    ######

    y_train_scaler = MinMaxScaler().fit(y_train)
    y_train = y_train_scaler.transform(y_train)

    y_val_scaler = MinMaxScaler().fit(y_val)

    ######

    X_train = np.concatenate(
        (X_train_mv, X_train_gpv),
        axis=1,
    )

    X_val = np.concatenate(
        (X_val_mv, X_val_gpv),
        axis=1,
    )

    return X_train, y_train, X_val, y_val_scaler


def denormalize_data(y_val_pred, y_val_scaler):
    """Inverse normalization function used for Scikit-learn regressors.

    Args:
        y_val_pred (_type_): the predictions for y_val set
        y_val_scaler (_type_): the y_val scaler

    Returns:
        _type_: the de-normalized y_val set
    """
    return y_val_scaler.inverse_transform(y_val_pred)


def normalization(tensor, norm_parameters):
    """Normalization function used for Keras NNs.
    MinMaxScaler is implemented.

    Args:
        tensor (_type_): the input tensor
        norm_parameters (_type_): a dictionary containing the parameters required for normalization

    Returns:
        _type_: the normalized tensor
    """
    size = tensor.get_shape()[1]
    normalized_tensor = tf.divide(
        tf.subtract(tensor, norm_parameters["min"][:size]),
        tf.subtract(
            norm_parameters["max"][:size],
            norm_parameters["min"][:size],
        ),
    )
    return replace_with_zeros(normalized_tensor)


def inverse_normalization(tensor, norm_parameters):
    """Inverse normalization function used for Keras NNs.

    Args:
        tensor (_type_): the input tensor
        norm_parameters (_type_): a dictionary containing the parameters required for de-normalization

    Returns:
        _type_: the de-normalized tensor
    """
    size = tensor.get_shape()[1]
    normalized_tensor = tf.add(
        tf.multiply(
            tensor,
            tf.subtract(
                norm_parameters["max"][-size:],
                norm_parameters["min"][-size:],
            ),
        ),
        norm_parameters["min"][-size:],
    )
    return replace_with_zeros(normalized_tensor)


def replace_with_zeros(normalized_tensor):
    # Replace NaN values with zeros
    return tf.where(
        tf.math.is_nan(normalized_tensor),
        tf.zeros_like(normalized_tensor),
        normalized_tensor,
    )
