import os
import subprocess
import utils.istarmap  # import to apply patch
from multiprocessing import Pool
from tqdm import tqdm


def create_directory(directory_path):
    """Create a directory.

    Args:
        directory_path (str, bytes or os.PathLike object): path to the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def create_commands(num_tasks):
    """Generate: the commands to run, the path where to save the std out, the path where to save the std error.

    Args:
        num_tasks (int): number of tasks to generate.

    Returns:
        list[tuple]: each element of the list contains a tuple with (cmd, std_out, std_err).
    """
    common_path = os.path.join("resources", "run_experiments_trial")
    create_directory(common_path)
    return [
        (
            f"python src/alternative_main.py --task_number {task_number}",
            os.path.join(common_path, f"std_out_{task_number}.txt"),
            os.path.join(common_path, f"std_err_{task_number}.txt"),
        )
        for task_number in range(1, num_tasks + 1)
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


# Variables for the example at hand
num_tasks = 24
pool_size = 8

# Generates the command to run
commands = create_commands(num_tasks)

# Create the progress bar (num_tasks to execute)
with tqdm(total=num_tasks) as pbar:
    # Create a pool of pool_size workers
    with Pool(pool_size) as pool:
        # Assign the commands (tasks) to the pool, and run it
        for _ in pool.istarmap(run_cmd, commands):
            pbar.update()
