import os
import re
import copy
import shutil
from collections import OrderedDict

algorithms = [
    "PersistentSystem",
    "LinearRegression",
    "RandomForest",
    "SVR",
    "FeedForward",
]

case_studies = [
    # Synthetic vs Synthetic
    {
        "field_names": {
            "train_field_name": "Synthetic field v.1.0",
            "val_field_name": "Synthetic field v.1.0",
            "test_field_name": "Synthetic field v.1.0",
        },
        "scenario_names": {
            "train_scenario_name": "Synthetic Martorano v.1.0",
            "val_scenario_name": "Synthetic Bologna v.1.0",
            "test_scenario_name": "Real Fondo PROGETTO_1 2020",  # real watering
        },
    },
    # Synthetic vs Real
    {
        "field_names": {
            "train_field_name": "Synthetic field v.1.0",
            "val_field_name": "Synthetic field v.1.0",
            "test_field_name": "Real Fondo PROGETTO_1",
        },
        "scenario_names": {
            "train_scenario_name": "Synthetic Martorano v.1.0",
            "val_scenario_name": "Synthetic Bologna v.1.0",
            "test_scenario_name": "Real Fondo PROGETTO_1 2020",  # real watering
        },
    },
    # Real vs Real
    {
        "field_names": {
            "train_field_name": "Real Fondo PROGETTO_2",
            "val_field_name": "Real Fondo PROGETTO_2",
            "test_field_name": "Real Fondo PROGETTO_1",
        },
        "scenario_names": {
            "train_scenario_name": "Real Fondo PROGETTO_2 2020",
            "val_scenario_name": "Real Fondo PROGETTO_2 2020",
            "test_scenario_name": "Real Fondo PROGETTO_1 2020",
        },
    },
]

rolling_window_parameters = [
    {
        "n_hours_ahead": value,
        "n_hours_past": value,
        "stride_ahead": value,
        "stride_past": value,
    }
    for value in [6, 12, 24, 48, 96, 128]
]

dict_template = OrderedDict(
    [
        (
            "field_names",
            {
                "train_field_name": "",
                "val_field_name": "",
                "test_field_name": "",
            },
        ),
        (
            "scenario_names",
            {
                "train_scenario_name": "",
                "val_scenario_name": "",
                "test_scenario_name": "",
            },
        ),
        ("arrangement", "Fondo PROGETTO"),
        (
            "window_parameters",
            {
                "n_hours_ahead": None,
                "n_hours_past": None,
                "stride_ahead": None,
                "stride_past": None,
            },
        ),
        (
            "tuning_parameters",
            {
                "algorithm_name": "",
                "kind": "",
                "description": "",
                "batch_size": 1,
                "seed": 42,
            },
        ),
    ]
)


def create_directory(directory_path):
    """Create a directory.

    Args:
        directory_path (str, bytes or os.PathLike object): path to the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def __write_run(path, run):
    try:
        print("   -> {}".format(path))
        with open(path, "w") as f:
            for k, v in run.items():
                if isinstance(v, str):
                    f.write("{}: {}\n".format(k, v))
                else:
                    f.write("{}:\n".format(k))
                    for i, j in v.items():
                        f.write("  {}: {}\n".format(i, j))
    except Exception as e:
        print(e)


def generate_runs():
    runs = []
    src_input_path = os.path.join("resources", "automl_inputs")

    for algorithm in algorithms:
        print("# DATASET: {}".format(algorithm))
        for case_study in case_studies:
            print("## CASE STUDY: {}".format(case_study))
            if case_study["field_names"][
                [*case_study["field_names"].keys()][-1]
            ].startswith("Synthetic"):
                case = "Synthetic vs Synthetic"
            elif case_study["field_names"][
                [*case_study["field_names"].keys()][0]
            ].startswith("Synthetic"):
                case = "Synthetic vs Real"
            else:
                case = "Real vs Real"

            for parameter in rolling_window_parameters:
                run = copy.deepcopy(dict_template)
                run["field_names"][
                    "train_field_name"
                ] = f"""{case_study[
                    'field_names'
                ]['train_field_name']}"""
                run["field_names"][
                    "val_field_name"
                ] = f"""{case_study[
                    "field_names"
                ]["val_field_name"]}"""
                run["field_names"][
                    "test_field_name"
                ] = f"""{case_study[
                    "field_names"
                ]["test_field_name"]}"""
                run["scenario_names"][
                    "train_scenario_name"
                ] = f"""{case_study[
                    "scenario_names"
                ]["train_scenario_name"]}"""
                run["scenario_names"][
                    "val_scenario_name"
                ] = f"""{case_study[
                    "scenario_names"
                ]["val_scenario_name"]}"""
                run["scenario_names"][
                    "test_scenario_name"
                ] = f"""{case_study[
                    "scenario_names"
                ]["test_scenario_name"]}"""
                run["window_parameters"]["n_hours_ahead"] = parameter["n_hours_ahead"]
                run["window_parameters"]["n_hours_past"] = parameter["n_hours_past"]
                run["window_parameters"]["stride_ahead"] = parameter["stride_ahead"]
                run["window_parameters"]["stride_past"] = parameter["stride_past"]
                run["tuning_parameters"][
                    "algorithm_name"
                ] = f"""{algorithm} v.1.0 HA {parameter["n_hours_ahead"]} HP {parameter["n_hours_past"]} SA {parameter["stride_ahead"]} SP {parameter["stride_past"]}"""
                run["tuning_parameters"]["kind"] = algorithm
                run["tuning_parameters"][
                    "description"
                ] = f'''"First attempt of {run["tuning_parameters"]["kind"]} with all sensors - Train field: {run["field_names"]["train_field_name"]}, Val field: {run["field_names"]["val_field_name"]}, Test field: {run["field_names"]["test_field_name"]} - Train scenario: {run["scenario_names"]["train_scenario_name"]}, Val scenario: {run["scenario_names"]["val_scenario_name"]}, Test scenario: {run["scenario_names"]["test_scenario_name"]} - Imputation: bfill+ffill - Normalization: standard"'''

                algo_name = re.sub(
                    " |\.", "_", run["tuning_parameters"]["algorithm_name"]
                )
                case_name = re.sub(" |\.", "_", case)
                outcomes_path = os.path.join("outcomes", f"run_{algo_name}_{case_name}")
                create_directory(os.path.join(outcomes_path, "logs"))
                create_directory(os.path.join(outcomes_path, "predictions"))
                __write_run(
                    os.path.join(
                        outcomes_path,
                        "config_{}_{}.yaml".format(algo_name, case_name),
                    ),
                    run,
                )

                runs.append(f"run_{algo_name}_{case_name}")

        # AutoML input files
        if algo_name == "PersistentSystem":
            shutil.copy2(
                os.path.join(src_input_path, "persistent_system_input.json"),
                os.path.join(outcomes_path, "automl_input.json"),
            )
        elif algo_name == "LinearRegression":
            shutil.copy2(
                os.path.join(src_input_path, "linear_regressor_input.json"),
                os.path.join(outcomes_path, "automl_input.json"),
            )
        elif algo_name == "RandomForest":
            shutil.copy2(
                os.path.join(src_input_path, "random_forest_input.json"),
                os.path.join(outcomes_path, "automl_input.json"),
            )
        elif algo_name == "SVR":
            shutil.copy2(
                os.path.join(src_input_path, "svr_input.json"),
                os.path.join(outcomes_path, "automl_input.json"),
            )
        else:
            shutil.copy2(
                os.path.join(src_input_path, "feed_forward_input.json"),
                os.path.join(outcomes_path, "automl_input.json"),
            )

    return runs
