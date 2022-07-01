import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from functools import partial
from flaml import tune

from utils.argparse import parse_args
from utils.json_to_csv import json_to_csv
from automl.data_acquisition import load_agro_data_from_csv, load_dataset_from_openml
from automl.optimization import objective
from automl.space_loading import get_space


def main(args):
    np.random.seed(args.seed)

    # Load the dataset
    X, y, _ = load_dataset_from_openml(args.dataset)
    # Load the space
    space = get_space(args.input_path)

    # Find best hyper-parameters
    start_time = time.time()
    analysis = tune.run(
        evaluation_function=partial(objective, X, y, args.metric, args.seed),
        config=space,
        metric=args.metric,
        mode=args.mode,
        num_samples=args.batch_size,
        time_budget_s=1800,
        verbose=0,
        max_failure=args.batch_size,
    )
    end_time = time.time()

    # Specify which information are needed for the output
    filtered_keys = [args.metric, "status", "config", "time_total_s"]
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

    # Convert the result in csv
    json_to_csv(automl_output=automl_output.copy(), args=args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
