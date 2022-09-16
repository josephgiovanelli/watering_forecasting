import os

import pandas as pd
import numpy as np

import math
import re

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

filtered_db_sensors = []  # sensors list


def load_agro_data_from_csv(path=os.path.join("/", "home", "data")):
    def _ground_potential_reconstruction(df):
        """Ground potential data likely contains missing values.
        Since this kind of data has to be treated properly, we define a custom imputation.
        Data from sensors are sampled every 15 minutes, but we are interested in hourly sampling.
        We are lucky if we have the data at the :00 of the hour at hand , bc we do not have to do nothing.
        Otherwise, we take the value of the previous quarter (15 minutes before).
        This is done until the minute :00 of the previous hour.

        Args:
            path (_type_): input data path

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
        "air_temperature",
        "air_humidity",
        "wind_speed",
        "solar_radiation",
        "precipitation",
        "irrigation",
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


def load_agro_data_from_db(db_address, db_port, db_user, db_password):
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
                    lambda x: pd.to_datetime(x["unix_timestamp"], unit="s").minute
                    == minute,
                    axis=1,
                )
            ]
            if minute != 0:
                dfs[minute]["unix_timestamp"] = (
                    (dfs[minute]["unix_timestamp"] + 3600) / 3600
                ).astype(int) * 3600
            dfs[minute] = dfs[minute].pivot(
                index="unix_timestamp", columns=["z", "y", "x"], values="value"
            )
        for minute in [45, 30, 15]:
            dfs[0] = dfs[0].combine_first(dfs[minute])
        return dfs[0]

    # DB connection
    engine = create_engine(
        "postgresql://{user}:{password}@{address}:{db_port}/smart_irrigation".format(
            user=db_user, password=db_password, address=db_address, db_port=db_port
        )
    )
    connection = engine.connect()

    # Queries
    arpae_query = "SELECT unix_timestamp, value_type_name, value \
        FROM synthetic_scenario_arpae_data \
        WHERE scenario_arpae_name IN (SELECT scenario_arpae_name \
		FROM synthetic_scenario \
		WHERE name = '{}') \
        ORDER BY unix_timestamp ASC, value_type_name ASC"

    water_query = "SELECT unix_timestamp, value_type_name, value \
        FROM synthetic_scenario_water_data \
        WHERE scenario_water_name IN (SELECT scenario_water_name \
		FROM synthetic_scenario \
		WHERE name = '{}') \
        ORDER BY unix_timestamp ASC, value_type_name ASC"

    gp_query = "SELECT unix_timestamp, ROUND((x - 1)::numeric, 3) AS x, ROUND((y - 1)::numeric, 3) AS y, z, value \
        FROM synthetic_data \
        WHERE scenario_name='{}' \
        ORDER BY unix_timestamp ASC, x ASC, y ASC, z DESC"

    sensors_query = "SELECT ROUND(x::numeric, 3) AS x, ROUND(z::numeric, 3) AS z \
        FROM synthetic_sensor \
        ORDER BY x ASC, z DESC"

    # Scenarios to consider
    scenarios = [
        "Real Martorano 2017 v.4.8",
        "Real Martorano 2018 v.4.8",
        "Real Martorano 2019 v.4.8",
        "Real Martorano 2020 v.4.8",
        "Real Bologna 2016 v.4.8",
        "Real Bologna 2017 v.4.8",
        "Real Bologna 2018 v.4.8",
        "Real Bologna 2019 v.4.8",
        "Real Fondo PROGETTO_1 v.4.8",
    ]

    # Get the scenarios data (weather data, watering data, ground potential data)
    for scenario in scenarios:
        scenario_name = re.sub(" |\.", "_", scenario.lower())
        locals()[scenario_name + "_arpae_df"] = pd.read_sql(
            arpae_query.format(scenario), connection
        )
        locals()[scenario_name + "_water_df"] = pd.read_sql(
            water_query.format(scenario), connection
        )
        locals()[scenario_name + "_gp_df"] = pd.read_sql(
            gp_query.format(scenario), connection
        )

        print(scenario)

    # Get the sensors coordinates
    sensors_coordinates_df = pd.read_sql(sensors_query, connection)
    sensors_coordinates_df.insert(
        loc=1, column="y", value=np.repeat(0.0, len(sensors_coordinates_df))
    )
    sensors = sensors_coordinates_df.to_numpy()

    # Get DB data sensors
    db_sensors = locals()["real_martorano_2017_v_4_8_gp_df"].loc[
        locals()["real_martorano_2017_v_4_8_gp_df"]["unix_timestamp"]
        == locals()["real_martorano_2017_v_4_8_gp_df"]["unix_timestamp"][0],
        "x":"z",
    ]
    db_sensors = db_sensors.to_numpy()

    # Filter sensors
    for sensor in sensors:
        min_diff = [math.inf, math.inf, math.inf]
        for db_sensor in db_sensors:
            diff = abs(sensor - db_sensor)
            if (diff <= min_diff).all() or np.isclose(diff, min_diff).all():
                min_diff = diff
                best_approx_db_sensor = db_sensor
        filtered_db_sensors.append(tuple(best_approx_db_sensor))

    # Pivot and filter data
    for scenario in scenarios:
        scenario_name = re.sub(" |\.", "_", scenario.lower())
        # Pivot weather and irrigation data
        locals()[scenario_name + "_arpae_df"] = locals()[
            scenario_name + "_arpae_df"
        ].pivot(index="unix_timestamp", columns="value_type_name", values="value")
        locals()[scenario_name + "_water_df"] = locals()[
            scenario_name + "_water_df"
        ].pivot(index="unix_timestamp", columns="value_type_name", values="value")

        # Filter ground potetntial data
        locals()[scenario_name + "_gp_df"] = locals()[scenario_name + "_gp_df"][
            [
                i in filtered_db_sensors
                for i in zip(
                    locals()[scenario_name + "_gp_df"].x,
                    locals()[scenario_name + "_gp_df"].y,
                    locals()[scenario_name + "_gp_df"].z,
                )
            ]
        ]
        # Pivot ground potetntial data
        locals()[scenario_name + "_gp_df"] = _ground_potential_reconstruction(
            locals()[scenario_name + "_gp_df"]
        )
        locals()[scenario_name + "_gp_df"].columns = [
            f"z{-column[0]}_y{column[1]}_x{column[2]}"
            for column in locals()[scenario_name + "_gp_df"].columns.values
        ]
        locals()[scenario_name + "_gp_df"] = locals()[scenario_name + "_gp_df"].reindex(
            sorted(locals()[scenario_name + "_gp_df"].columns, reverse=True), axis=1
        )

        # Concatenate weather, watering and ground potential data
        locals()[scenario_name + "_df"] = pd.concat(
            [
                locals()[scenario_name + "_arpae_df"],
                locals()[scenario_name + "_water_df"],
                locals()[scenario_name + "_gp_df"],
            ],
            axis=1,
        )

        print(scenario, " - Pivoted")

    # Create train, validation and test sets
    train = pd.concat(
        (
            locals()["real_martorano_2017_v_4_8_df"],
            locals()["real_martorano_2018_v_4_8_df"],
            locals()["real_martorano_2019_v_4_8_df"],
            locals()["real_martorano_2020_v_4_8_df"],
        ),
        axis=0,
    )
    val = pd.concat(
        (
            locals()["real_bologna_2016_v_4_8_df"],
            locals()["real_bologna_2017_v_4_8_df"],
            locals()["real_bologna_2018_v_4_8_df"],
            locals()["real_bologna_2019_v_4_8_df"],
        ),
        axis=0,
    )
    test = locals()["real_fondo_progetto_1_v_4_8_df"]

    for dataset in ["train", "val", "test"]:
        # Imputation
        locals()[dataset] = (
            locals()[dataset].fillna(method="bfill").fillna(method="ffill")
        )
        # Rename columns
        locals()[dataset].index.names = ["timestamp"]
        locals()[dataset].rename(
            columns={
                "AIR_HUMIDITY": "air_humidity",
                "AIR_TEMPERATURE": "air_temperature",
                "RADIATIONS": "solar_radiation",
                "WIND_SPEED": "wind_speed",
                "IRRIGATIONS": "irrigation",
                "PRECIPITATIONS": "precipitation",
            },
            inplace=True,
        )
        # Reorder columns
        cols = [
            "air_temperature",
            "air_humidity",
            "wind_speed",
            "solar_radiation",
            "precipitation",
            "irrigation",
        ] + list(locals()[dataset].columns[6:].values)
        locals()[dataset] = locals()[dataset][cols]

    return train, val, test


def compute_statistics(window_size):
    """Set min and max values for each variable.

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
    max_gp_value = [0.0]

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


def create_rolling_window(df, window_size, stride=1):
    """Create the rolling window, according to the specified window size and stride.

    Args:
        df (_type_): input dataset
        window_size (_type_): the specified window size
        stride (_type_, optional): the specified window stride. Defaults to 1.

    Returns:
        _type_: rolling window dataset
    """
    if window_size < 2 or (stride < 1):
        print(
            "The window size and the stride must be greater than 1 and 0 respectively!"
        )
    else:
        m_var_names = list(
            df.columns[:4]
        )  # air_temperature, air_humidity, wind_speed, solar_radiation
        w_var_names = list(df.columns[4:6])  # precipitation, irrigation
        gp_var_names = list(df.columns[6:])  # ground potential

        if stride > 1:
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


# To be used with "local" data
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
    X = data.iloc[:, : -output_horizon * 12]
    y = data.iloc[:, -output_horizon * 12 :]

    X_val = []
    y_val = []

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y.to_numpy(),
        test_size=test_ratio,
        random_state=42,
        shuffle=shuffle,
    )

    if val_ratio != None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=42, shuffle=shuffle
        )

    y_train = pd.DataFrame(
        y_train, columns=y.columns, index=y.iloc[: len(y_train), :].index
    )
    y_val = pd.DataFrame(
        y_val,
        columns=y.columns,
        index=y.iloc[len(y_train) : len(y_train) + len(y_val), :].index,
    )
    y_test = pd.DataFrame(
        y_test, columns=y.columns, index=y.iloc[-len(y_test) :, :].index
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_labels(train_data, val_data, test_data, output_horizon):
    """Split the given datasets to obtain the samples and the corresponding labels.

    Args:
        train_data (_type_): the training data
        val_data (_type_): the validation data
        test_data (_type_): the test data
        output_horizon (_type_): the specified output horizon in hours

    Returns:
        _type_: X_train is the training set, y_train is the training label set,
                X_val is the validation set, y_val is the validation label set,
                X_test is the test set, y_test is the test label set
    """
    X_train = train_data.iloc[:, : -output_horizon * 12]
    y_train = train_data.iloc[:, -output_horizon * 12 :]

    X_val = val_data.iloc[:, : -output_horizon * 12]
    y_val = val_data.iloc[:, -output_horizon * 12 :]

    X_test = test_data.iloc[:, : -output_horizon * 12]
    y_test = test_data.iloc[:, -output_horizon * 12 :]

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X_train, y_train, X_val, y_val, X_test, y_test, window_size):
    """Normalization function used for Scikit-learn regressors.
    MinMaxScaler is used.

    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        y_val (_type_): the validation label set
        X_test (_type_): the test set
        y_test (_type_): the test label set
        window_size (_type_): the specified window size

    Returns:
        _type_: the normalized X_train, y_train, X_val and X_test sets,
                the y_train_scaler, y_val scaler and y_test_scaler scalers
    """
    X_train_mv_scaler = MinMaxScaler().fit(X_train[:, : window_size * 6])
    X_train_mv = X_train_mv_scaler.transform(X_train[:, : window_size * 6])

    X_train_gpv_scaler = MinMaxScaler().fit(X_train[:, window_size * 6 :])
    X_train_gpv = X_train_gpv_scaler.transform(X_train[:, window_size * 6 :])

    X_val_mv_scaler = MinMaxScaler().fit(X_val[:, : window_size * 6])
    X_val_mv = X_val_mv_scaler.transform(X_val[:, : window_size * 6])

    X_val_gpv_scaler = MinMaxScaler().fit(X_val[:, window_size * 6 :])
    X_val_gpv = X_val_gpv_scaler.transform(X_val[:, window_size * 6 :])

    X_test_mv_scaler = MinMaxScaler().fit(X_test[:, : window_size * 6])
    X_test_mv = X_test_mv_scaler.transform(X_test[:, : window_size * 6])

    X_test_gpv_scaler = MinMaxScaler().fit(X_test[:, window_size * 6 :])
    X_test_gpv = X_test_gpv_scaler.transform(X_test[:, window_size * 6 :])

    ######

    y_train_scaler = MinMaxScaler().fit(y_train)
    y_train = y_train_scaler.transform(y_train)

    y_val_scaler = MinMaxScaler().fit(y_val)

    y_test_scaler = MinMaxScaler().fit(y_test)

    ######

    X_train = np.concatenate(
        (X_train_mv, X_train_gpv),
        axis=1,
    )

    X_val = np.concatenate(
        (X_val_mv, X_val_gpv),
        axis=1,
    )

    X_test = np.concatenate(
        (X_test_mv, X_test_gpv),
        axis=1,
    )

    return X_train, y_train, y_train_scaler, X_val, y_val_scaler, X_test, y_test_scaler


def denormalize_data(
    y_train_pred, y_train_scaler, y_val_pred, y_val_scaler, y_test_pred, y_test_scaler
):
    """Inverse normalization function used for Scikit-learn regressors.

    Args:
        y_train_pred (_type_): the predictions for y_train set
        y_train_scaler (_type_): the y_train scaler
        y_val_pred (_type_): the predictions for y_val set
        y_val_scaler (_type_): the y_val scaler
        y_test_pred (_type_): the predictions for y_test set
        y_test_scaler (_type_): the y_test scaler

    Returns:
        _type_: the de-normalized y_train, y_val and y_test sets
    """
    return (
        y_train_scaler.inverse_transform(y_train_pred),
        y_val_scaler.inverse_transform(y_val_pred),
        y_test_scaler.inverse_transform(y_test_pred),
    )


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


def plot_results(best_predictions, ground_truth, output_horizon, set_name, statistics):
    """Plot the best configuration result for each sensor.

    Args:
        best_predictions (_type_): the predictions of the best configuration found
        ground_truth (_type_): the ground truth
        output_horizon (_type_): the specified output horizon in hours
        set_name (_type_): the set to consider for the plot
        statistics (_type_): the dictionary containing dataset statistics
    """
    best_predictions = best_predictions.add_suffix("_pred")
    best_predictions.set_index("timestamp_pred", inplace=True)

    result = pd.concat([best_predictions, ground_truth], axis=1, join="inner")

    # Extract hours ahead considered in the predictions
    hours_ahead = set()
    for elem in ground_truth.columns:
        hours_ahead.add(elem[elem.rindex("_") + 1 :])
    hours_ahead = list(hours_ahead)
    hours_ahead.sort()

    # Extract sensors coordinates
    global filtered_db_sensors
    x_coords = list(set(np.array(filtered_db_sensors)[:, 0]))
    x_coords.sort()
    y_coords = list(set(np.array(filtered_db_sensors)[:, 1]))
    y_coords.sort()
    z_coords = list(set(np.array(filtered_db_sensors)[:, 2] * -1))
    z_coords.sort()

    # Create plot
    nrows = 3
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * output_horizon)
    fig.set_size_inches(24, 12, forward=True)
    for idh, h in enumerate(hours_ahead):
        for idz, z in enumerate(z_coords):
            for _, y in enumerate(y_coords):
                for idx, x in enumerate(x_coords):
                    result[[f"z{z}_y{y}_x{x}_{h}", f"z{z}_y{y}_x{x}_{h}_pred"]].plot(
                        ax=axes[idz, idx + (ncols * idh)]
                    )
                    axes[idz, idx + (ncols * idh)].set_title(
                        f"z{z}_y{y}_x{x} - {h} h ahead"
                    )
                    axes[idz, idx + (ncols * idh)].set_ylim(
                        [
                            int(statistics["min"][-1].numpy()),
                            int(statistics["max"][-1].numpy()),
                        ]
                    )

    plt.savefig(os.path.join("/", "home", "resources", f"{set_name}_best_results.png"))
