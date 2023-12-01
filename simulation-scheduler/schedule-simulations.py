import argparse
import os
import re
import subprocess
from datetime import datetime
import time

# Constants
SLURM_MAX_JOBS = 100
SEPARATOR = "============================================================"


def check_arguments(args, parser):
    if not args.script_to_start.strip():
        parser.error("script_to_start argument is empty")
    if not args.directory_to_check.strip():
        parser.error("directory_to_check argument is empty")
    if not (args.total_simulations_number >= 1):
        parser.error("total_simulations_number must be >= 1")
    if not (args.minutes_between_checks >= 1):
        parser.error("minutes_between_checks must be >= 1")


def check_files_and_directories_exist(args):
    if not os.path.isdir(args.directory_to_check):
        raise FileNotFoundError(f"The directory specified in directory_to_check does not exist: {args.directory_to_check}")
    if not os.path.isfile(args.script_to_start):
        raise FileNotFoundError(f"The file specified in script_to_start does not exist: {args.script_to_start}")


def extract_simulation_number(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Search for the pattern --simulations_number= followed by numbers
                match = re.search(r'--simulations_number=(\d+)', line)
                if match:
                    # Return the first matched group, which is the number
                    return int(match.group(1))
    except IOError as e:
        raise IOError(f"simulations_number was not found in the {file_path}")


def can_fire_next_batch_squeue(simulations_number: int) -> bool:
    result = subprocess.run(["squeue"], capture_output=True, text=True, check=True)
    lines = result.stdout.split("\n")
    non_empty_lines_count = sum(1 for line in lines if line.strip())
    active_jobs_number: int = non_empty_lines_count - 1  # without header
    print(f"Currently running jobs: {active_jobs_number}")
    if simulations_number + active_jobs_number <= SLURM_MAX_JOBS:
        print(f"[{simulations_number}] new simulations can be scheduled {simulations_number} + {active_jobs_number} <= {SLURM_MAX_JOBS}")
        return True
    else:
        print(f"[{simulations_number}] new simulations can not be scheduled {simulations_number} + {active_jobs_number} > {SLURM_MAX_JOBS}")
        return False


def total_simulations_number_present(directory_to_check: str) -> int:
    directories_number = sum(os.path.isdir(os.path.join(directory_to_check, d)) for d in os.listdir(directory_to_check))
    return directories_number


def fire_next_batch(script_to_start: str) -> bool:
    result = subprocess.run(script_to_start, shell=True)
    if result.returncode != 0:
        print(f"Script {script_to_start} failed to execute. Return code [{result.returncode}]")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0


def now() -> str:
    current_timestamp = datetime.now()
    return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description='This tool starts executes a shell script defined in --script_to_start (which should start N<=100 slurm jobs).\n'
                                                 'Then it waits, checking squeue for a free slots for next N<=100 jobs.\n'
                                                 'Process terminates, when --directory_to_check contains contains the --total_simulations_number of simulations.\n'
                                                 'Start this tool with nohup and in background (with &) to prevent the termination on logout. \n'
                                                 'Process waits --minutes_between_checks until performing the next squeue check',
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Define the expected arguments
    parser.add_argument('--script_to_start', type=str, required=True,
                        help='Script that contains call of start-simulations.py')
    parser.add_argument('--directory_to_check', type=str, required=True,
                        help='Directory with simulation results (normally the root directory used in --script_to_start). Directory should be created before execution')
    parser.add_argument('--total_simulations_number', type=int, required=True,
                        help='Number of directories expected to be present at the end in --directory_to_check')
    parser.add_argument('--minutes_between_checks', type=int, required=True,
                        help='Minutes between scheduler checks')

    # Parse the arguments
    args = parser.parse_args()

    # Check the arguments
    check_arguments(args, parser)

    # Check the paths
    check_files_and_directories_exist(args)

    simulations_in_one_batch = extract_simulation_number(args.script_to_start)

    print(f"{now()} Starting a scheduler with following parameters:")
    print(f"script_to_start: {args.script_to_start}")
    print(f"directory_to_check: {args.directory_to_check}")
    print(f"simulations_in_one_batch: {simulations_in_one_batch}")
    print(f"total_simulations_number: {args.total_simulations_number}")
    print(f"minutes_between_checks: {args.minutes_between_checks}")

    while True:
        simulations_present = total_simulations_number_present(args.directory_to_check)
        if simulations_present >= args.total_simulations_number:
            print(f"{args.total_simulations_number} are present in directory {args.directory_to_check}")
            print("SUCCESS")
            exit(0)
        else:
            if can_fire_next_batch_squeue(simulations_in_one_batch):
                batch_started_successfully = fire_next_batch(args.script_to_start)
                if batch_started_successfully:
                    print(
                        f"{now()}: Started {simulations_in_one_batch} new simulations. [{total_simulations_number_present(args.directory_to_check)}/{args.total_simulations_number}]")
                else:
                    print(f"{now()}: There was a problem starting a batch.")
            else:
                print(f"{now()}: Can not fire new batch now.")
                # wait if simulations were not started
                seconds_to_sleep = args.minutes_between_checks * 60
                print(f"{now()}: Try to schedule new jobs in [{args.minutes_between_checks}] minute(s)")
                print(SEPARATOR)
                time.sleep(seconds_to_sleep)


if __name__ == "__main__":
    main()
