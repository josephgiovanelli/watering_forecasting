import os
import subprocess
import utils.istarmap  # import to apply patch
from multiprocessing import Pool
from tqdm import tqdm

from run_generator import generate_runs


def create_commands(runs):
    """Generate: the commands to run, the path where to save the std out, the path where to save the std err.

    Args:
        runs (int): the runs to consider.

    Returns:
        list[tuple]: each element of the list contains a tuple with (cmd, std_out, std_err).
    """
    return [
        (
            f"""python src/main.py --run_directory_path {os.path.join("outcomes", run)} --config_file_path {os.path.join("outcomes", run, f"config_{run.split('_', 1)[1]}.yaml")} --db_credentials_file_path {os.path.join("resources", "db_credentials.yaml")}""",
            os.path.join("outcomes", run, "logs", f"std_out.txt"),
            os.path.join("outcomes", run, "logs", f"std_err.txt"),
        )
        for run in runs
    ]


def run_cmd(cmd, stdout_path, stderr_path):
    """Run a command in the shell, and save std out and std err.

    Args:
        cmd (string): the command to run
        stdout_path (str, bytes or os.PathLike object): where to save the std out.
        stderr_path (str, bytes or os.PathLike object): where to save the std err.
    """
    open(stdout_path, "w")
    open(stderr_path, "w")
    with open(stdout_path, "a") as log_out:
        with open(stderr_path, "a") as log_err:
            subprocess.call(cmd, stdout=log_out, stderr=log_err, bufsize=0, shell=True)


# Generate the runs to consider
runs = generate_runs()

# Variables for the example at hand
num_tasks = len(runs)
pool_size = 8

# Generate the command to run
commands = create_commands(runs)

# Create the progress bar (num_tasks to execute)
with tqdm(total=num_tasks) as pbar:
    # Create a pool of pool_size workers
    with Pool(pool_size) as pool:
        # Assign the commands (tasks) to the pool, and run it
        for _ in pool.istarmap(run_cmd, commands):
            pbar.update()
