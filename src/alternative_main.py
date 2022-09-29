import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Trial")
    parser.add_argument(
        "-task_number",
        "--task_number",
        nargs="?",
        type=int,
        required=True,
        help="no. of the executed task",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    time.sleep(10)
    print(args.task_number)
