import os
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from functools import partial
from flaml import tune

from utils.argparse import parse_args
from utils.data_acquisition import (
    load_agro_data_from_csv,
    load_agro_data_from_db,
    create_rolling_window,
    train_val_test_split,
    get_data_labels,
    compute_statistics,
    plot_results,
)
from automl.optimization import objective
from automl.space_loading import get_space


def main(args):
    np.random.seed(args.seed)

    # Load the datasets
    train_data, val_data, test_data = load_agro_data_from_db(
        args.db_address, args.db_port, args.db_user, args.db_password
    )

    # Create rolling windows
    rolling_window_train_data = create_rolling_window(
        train_data, args.window_size, args.stride
    )
    rolling_window_val_data = create_rolling_window(
        val_data, args.window_size, args.stride
    )
    rolling_window_test_data = create_rolling_window(
        test_data, args.window_size, args.stride
    )

    # Get data labels
    X_train, y_train, X_val, y_val, X_test, y_test = get_data_labels(
        rolling_window_train_data,
        rolling_window_val_data,
        rolling_window_test_data,
        args.output_horizon,
    )

    # Compute statistics on the dataset
    statistics = compute_statistics(args.window_size)

    # Load the space
    space = get_space(args.input_path)

    # Find best hyper-parameters
    start_time = time.time()
    analysis = tune.run(
        evaluation_function=partial(
            objective,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args.window_size,
            args.output_horizon,
            statistics,
            args.seed,
        ),
        config=space,
        metric="val_score",
        mode="min",
        num_samples=args.batch_size,
        time_budget_s=1800,
        verbose=0,
        max_failure=args.batch_size,
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
    with open(args.output_path, "w") as outfile:
        json.dump(automl_output, outfile)

    # Plot best result for each set
    for set in ["train", "val", "test"]:
        plot_results(
            pd.read_csv(
                os.path.join(
                    "/",
                    "home",
                    "resources",
                    "predictions",
                    f"""conf_{automl_output["best_config"]["conf"]}_{set}.csv""",
                )
            ),
            locals()[f"y_{set}"],
            args.output_horizon,
            set,
            statistics,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
