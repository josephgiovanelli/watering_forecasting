import os
import json
import time
import warnings

from utils.parameters import Parameters

warnings.filterwarnings("ignore")

import pandas as pd

pd.set_option("display.max_columns", 500)
# pd.set_option("display.max_rows", 5000)

import numpy as np

from functools import partial
from flaml import tune

from utils.parameter_parsing import parse_args, load_conf
from utils.data_acquisition import (
    load_agro_data_from_csv,
    open_db_connection,
    close_db_connection,
    load_agro_data_from_db,
    get_data_labels,
    train_val_test_split,
    plot_results,
    create_rolling_window,
)
from automl.optimization import objective
from automl.space_loading import get_space


def main(args, run_cfg, db_cfg):
    np.random.seed(run_cfg["tuning_parameters"]["seed"])

    """
    ### LOCAL MODE
    df = load_agro_data_from_csv()
    df = create_rolling_window(df, run_cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        df, 1, test_ratio=0.33, val_ratio=0.5, shuffle=False
    )
    """

    ### DB MODE
    # Load the datasets from DB and create the rolling windows
    train_data, val_data, test_data = load_agro_data_from_db(run_cfg, db_cfg)

    # Get data labels
    X_train, y_train, X_val, y_val, X_test, y_test = get_data_labels(
        train_data, val_data, test_data
    )

    # Load the space
    space = get_space(os.path.join(args.run_directory_path, "automl_input.json"))

    ### DB MODE
    connection = open_db_connection(db_cfg)

    # Get the name of the original sensors
    sensors_name_query = "SELECT sensor_name \
        FROM synthetic_sensor_arrangement \
        WHERE arrangement_name = '{}' \
        ORDER BY sensor_name::numeric ASC"
    sensors_name_df = pd.read_sql(
        sensors_name_query.format(run_cfg["arrangement"]),
        connection,
    )
    sensors_name_list = sensors_name_df.values.flatten()

    # Find best hyper-parameters
    start_time = time.time()
    analysis = tune.run(
        evaluation_function=partial(
            objective,
            X_train.copy(),
            y_train.copy(),
            X_val.copy(),
            y_val.copy(),
            X_test.copy(),
            y_test.copy(),
            run_cfg["window_parameters"]["stride_past"],
            run_cfg["tuning_parameters"]["seed"],
            args.run_directory_path,
            sensors_name_list,
        ),
        config=space,
        metric="val_score",
        mode="min",
        num_samples=run_cfg["tuning_parameters"]["batch_size"],
        time_budget_s=1800,
        verbose=0,
        max_failure=run_cfg["tuning_parameters"]["batch_size"],  #
    )
    end_time = time.time()

    # Specify which information are needed for the output
    filtered_keys = [
        "train_raw_scores",
        "val_raw_scores",
        "test_raw_scores",
        "train_score",
        "val_score",
        "test_score",
        "status",
        "conf",
        "config",
        "time_total_s",
    ]
    # Prepare the output file
    automl_output = {
        "optimization_time": end_time - start_time,
        # Filter the information for the best config
        "best_config": {
            key: value
            for key, value in analysis.best_trial.last_result.items()
            if key in filtered_keys
        },
        # For each visited config, filter the information
        "results": [
            {
                key: value if value != float("-inf") else str(value)
                for key, value in values.items()
                if key in filtered_keys
            }
            for values in analysis.results.values()
        ],
    }

    # Export the result
    with open(
        os.path.join(args.run_directory_path, "automl_output.json"), "w"
    ) as outfile:
        json.dump(automl_output, outfile)

    sensor_name_query = "SELECT name FROM synthetic_sensor \
        WHERE x={x} AND y={y} AND z={z}"

    humidity_bins_query = "SELECT * FROM synthetic_humidity_bins \
            ORDER BY min ASC"
    humidity_bins_df = pd.read_sql(
        humidity_bins_query, connection, index_col="humidity_bin"
    )

    humidity_bins_count_dict = {bin: 0 for bin in humidity_bins_df.index}

    common_col_dict = {
        "algorithm_name": run_cfg["tuning_parameters"]["algorithm_name"],
        "train_field_name": run_cfg["field_names"]["train_field_name"],
        "train_scenario_name": run_cfg["scenario_names"]["train_scenario_name"],
        "re_training": "NONE",
        "re_training_timestamp": "NONE",
    }

    basic_col_dict = {
        "n_hour_past": run_cfg["window_parameters"]["n_hours_past"],
        "n_hour_ahead": run_cfg["window_parameters"]["n_hours_ahead"],
        "stride_past": run_cfg["window_parameters"]["stride_past"],
        "stride_ahead": run_cfg["window_parameters"]["stride_ahead"],
        "recursion": "false",
        "pca": "false",
        "hyperparameter_tuning": "true",
        "normalization": "STANDARD",
    }

    # Populate 'synthetic_algorithm' table
    algo_col_dict = dict(basic_col_dict)
    algo_col_dict.update(
        {
            hp: automl_output["best_config"]["config"]["regression"][hp]
            for hp in automl_output["best_config"]["config"]["regression"]
            if hp != "meta_estimator"
            and hp != "super_type"
            and hp != "type"
            and hp != "base_estimator"
            and hp != "n_jobs"
        }
    )
    algo_col_dict["name"] = run_cfg["tuning_parameters"]["algorithm_name"]
    algo_col_dict["kind"] = run_cfg["tuning_parameters"]["kind"]
    algo_col_dict["description"] = run_cfg["tuning_parameters"]["description"]
    algo_df = pd.DataFrame.from_dict([algo_col_dict])

    # Populate 'synthetic_algorithm_hyperparameters' table
    syn_hyper_list = []
    for elem in basic_col_dict:
        hyper_col_dict = {
            "algorithm_name": run_cfg["tuning_parameters"]["algorithm_name"],
            "hyperparameter_name": elem,
            "value": basic_col_dict[elem],
        }
        syn_hyper_list.append(hyper_col_dict)
    for hp in automl_output["best_config"]["config"]["regression"]:
        if (
            hp != "meta_estimator"
            and hp != "super_type"
            and hp != "type"
            and hp != "n_jobs"
        ):
            hyper_col_dict = {
                "algorithm_name": run_cfg["tuning_parameters"]["algorithm_name"],
                "hyperparameter_name": hp,
                "value": automl_output["best_config"]["config"]["regression"][hp],
            }
            syn_hyper_list.append(hyper_col_dict)

    hyper_df = pd.DataFrame(syn_hyper_list)

    # print("algo_df")
    # print(algo_df)
    # print("hyper_df")
    # print(hyper_df)

    algo_df.to_sql(
        name="synthetic_algorithm", con=connection, index=False, if_exists="append"
    )
    hyper_df.to_sql(
        name="synthetic_algorithm_hyperparamaters",
        con=connection,
        index=False,
        if_exists="append",
    )

    for set in ["train", "val", "test"]:
        best_pred_df = pd.read_csv(
            os.path.join(
                args.run_directory_path,
                "predictions",
                f"""conf_{automl_output["best_config"]["conf"]}_{set}.csv""",
            ),
            index_col="unix_timestamp",
        )

        common_col_dict["field_name"] = run_cfg["field_names"][f"{set}_field_name"]
        common_col_dict["scenario_name"] = run_cfg["scenario_names"][
            f"{set}_scenario_name"
        ]
        common_col_dict["dataset"] = set

        syn_pred_col_dict = dict(common_col_dict)
        syn_pred_col_dict["value_type_name"] = "GROUND_WATER_POTENTIAL"
        syn_pred_col_dict[
            "prediction_type_name"
        ] = f"""{run_cfg["window_parameters"]["n_hours_ahead"]}h AHEAD"""

        syn_pred_rows_list = []
        syn_pred_hum_bins_rows_list = []
        syn_sensor_rows_list = []

        for x_coord in Parameters().get_original_x_coords():
            for y_coord in Parameters().get_original_y_coords():
                for z_coord in Parameters().get_original_z_coords():
                    for idx, _ in best_pred_df.iterrows():
                        # Populate 'synthetic_prediction' table
                        new_syn_pred_col_dict = dict(syn_pred_col_dict)
                        new_syn_pred_col_dict["unix_timestamp"] = idx
                        new_syn_pred_col_dict["x"] = x_coord
                        new_syn_pred_col_dict["y"] = y_coord
                        new_syn_pred_col_dict["z"] = z_coord
                        new_syn_pred_col_dict["value"] = best_pred_df.loc[
                            idx,
                            f"""z{z_coord}_y{y_coord}_x{x_coord}_a{run_cfg["window_parameters"]["n_hours_ahead"]}""",
                        ]
                        new_syn_pred_col_dict["original_value"] = locals()[
                            f"y_{set}"
                        ].loc[
                            idx,
                            f"""z{z_coord}_y{y_coord}_x{x_coord}_a{run_cfg["window_parameters"]["n_hours_ahead"]}""",
                        ]
                        new_syn_pred_col_dict[
                            f"""value_{run_cfg["window_parameters"]["n_hours_ahead"]}"""
                        ] = best_pred_df.loc[
                            idx,
                            f"""z{z_coord}_y{y_coord}_x{x_coord}_a{run_cfg["window_parameters"]["n_hours_ahead"]}""",
                        ]
                        syn_pred_rows_list.append(new_syn_pred_col_dict)

                    # Populate 'synthetic_field_scenario_algorithm_sensor' table
                    syn_sensor_col_dict = dict(common_col_dict)
                    sensor_name_df = pd.read_sql(
                        sensor_name_query.format(x=x_coord, y=y_coord, z=z_coord),
                        connection,
                    )
                    sensor_name = sensor_name_df.values.item()
                    syn_sensor_col_dict["sensor_name"] = sensor_name
                    syn_sensor_col_dict["rmse"] = automl_output["best_config"][
                        f"{set}_raw_scores"
                    ][f"{set}_raw_score_{sensor_name}"]
                    syn_sensor_col_dict[
                        f"""rmse_{run_cfg["window_parameters"]["n_hours_ahead"]}"""
                    ] = automl_output["best_config"][f"{set}_raw_scores"][
                        f"{set}_raw_score_{sensor_name}"
                    ]
                    syn_sensor_rows_list.append(syn_sensor_col_dict)

        common_col_dict[
            "prediction_type_name"
        ] = f"""{run_cfg["window_parameters"]["n_hours_ahead"]}h AHEAD"""

        # Populate 'synthetic_prediction_humidity_bins' table
        for idx, _ in best_pred_df.iterrows():
            bins_count_dict = dict(humidity_bins_count_dict)
            syn_pred_hum_bins_col_dict = dict(common_col_dict)
            syn_pred_hum_bins_col_dict["unix_timestamp"] = idx
            # Compute bins count
            for coord in Parameters().get_original_sensor_columns():
                for index, col in humidity_bins_df.iterrows():
                    if (
                        best_pred_df.loc[
                            idx,
                            f"""{coord}_a{run_cfg["window_parameters"]["n_hours_ahead"]}""",
                        ]
                        >= col["min"]
                        and best_pred_df.loc[
                            idx,
                            f"""{coord}_a{run_cfg["window_parameters"]["n_hours_ahead"]}""",
                        ]
                        < col["max"]
                    ):
                        bins_count_dict[index] += 1
            for bin in bins_count_dict:
                new_syn_pred_hum_bins_col_dict = dict(syn_pred_hum_bins_col_dict)
                new_syn_pred_hum_bins_col_dict["humidity_bin"] = bin
                new_syn_pred_hum_bins_col_dict["count"] = bins_count_dict[bin]
                syn_pred_hum_bins_rows_list.append(new_syn_pred_hum_bins_col_dict)

        del common_col_dict["prediction_type_name"]

        # Populate 'synthetic_field_scenario_algorithm' table
        syn_algo_col_dict = dict(common_col_dict)
        syn_algo_col_dict["rmse"] = automl_output["best_config"][f"{set}_score"]
        syn_algo_col_dict[
            f"""rmse_{run_cfg["window_parameters"]["n_hours_ahead"]}"""
        ] = automl_output["best_config"][f"{set}_score"]
        syn_algo_df = pd.DataFrame.from_dict([syn_algo_col_dict])

        syn_pred_df = pd.DataFrame(syn_pred_rows_list)
        syn_pred_hum_bins_df = pd.DataFrame(syn_pred_hum_bins_rows_list)
        syn_sensor_df = pd.DataFrame(syn_sensor_rows_list)

        # print("syn_pred_df")
        # print(syn_pred_df)
        # print("syn_pred_hum_bins_df")
        # print(syn_pred_hum_bins_df)
        # print("syn_sensor_df")
        # print(syn_sensor_df)
        # print("syn_algo_df")
        # print(syn_algo_df)

        # Store best result on DB
        syn_pred_hum_bins_df.to_sql(
            name="synthetic_prediction_humidity_bins",
            con=connection,
            index=False,
            if_exists="append",
        )
        syn_sensor_df.to_sql(
            name="synthetic_field_scenario_algorithm_sensor",
            con=connection,
            index=False,
            if_exists="append",
        )
        syn_algo_df.to_sql(
            name="synthetic_field_scenario_algorithm",
            con=connection,
            index=False,
            if_exists="append",
        )
        syn_pred_df.to_sql(
            name="synthetic_prediction", con=connection, index=False, if_exists="append"
        )

        # Plot best result for each set
        plot_results(
            best_pred_df,
            locals()[f"y_{set}"],
            set,
            run_cfg["window_parameters"]["n_hours_ahead"],
            args.run_directory_path,
        )

    ### DB MODE
    close_db_connection(connection)


if __name__ == "__main__":
    args = parse_args()
    run_cfg = load_conf(args.config_file_path)
    db_cfg = load_conf(args.db_credentials_file_path)
    main(args, run_cfg, db_cfg)
