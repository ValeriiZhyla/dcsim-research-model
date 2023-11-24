import argparse
import os
import datetime
import uuid

DCSIM_EXECUTABLE_PATH = "~/dc-sim"
SEPARATOR = "=============================================="

def check_arguments(args, parser):
    if not (0 >= args.max_simulation_duration_minutes < 60):
        parser.error("max_simulation_duration_minutes must contain a value in interval [0,60)")
    if not (0 >= args.max_simulation_duration_hours):
        parser.error("max_simulation_duration_minutes must be >= 0")
    if not args.platform_file.strip():
        parser.error("platform_file argument is empty")
    if not args.workload_file.strip():
        parser.error("workload_file argument is empty")
    if not args.dataset_file.strip():
        parser.error("dataset_file argument is empty")
    if not (args.simulations_number >= 1):
        parser.error("simulations_number must be >= 1")


def check_files_exist(args):
    if not os.path.isfile(args.platform_file):
        raise FileNotFoundError(f"The file specified in platform_file does not exist: {args.platform_file}")
    if not os.path.isfile(args.workload_file):
        raise FileNotFoundError(f"The file specified in workload_file does not exist: {args.workload_file}")
    if not os.path.isfile(args.dataset_file):
        raise FileNotFoundError(f"The file specified in dataset_file does not exist: {args.dataset_file}")


def check_dcsim_exist():
    if not os.path.isfile(DCSIM_EXECUTABLE_PATH):
        raise FileNotFoundError(f"The DCSim executable does not exist: {DCSIM_EXECUTABLE_PATH}")


def format_time(hours, minutes):
    return f"{hours:02}:{minutes:02}"


def generate_uuids(n):
    uuids = []
    for _ in range(n):
        # Get the current date
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        # Generate a unique identifier
        unique_id = f"{date_str}-{uuid.uuid4()}"
        uuids.append(unique_id)
    return uuids


def prepare_simulations(simulation_uuids):
    for simulation_id in simulation_uuids:
        create_directory()
        move_simulation_files()
        create_slurm_job_file()
        create_metadata_file()
    print("Simulation directories prepared")
    print(SEPARATOR)

def check_integrity(simulation_uuids):
    check_all_directories_exist()
    check_each_directory_contains_slurm_file()
    check_each_directory_contains_metadata_file()
    check_each_directory_contains_dcsim_configurations()
    print("Checked the integrity of simulation directories")
    print(SEPARATOR)


def start_simulations(simulation_uuids):
    for simulation_id in simulation_uuids:
        start_simulation(simulation_id)
    print("Simulations started")
    print(SEPARATOR)


def main():
    parser = argparse.ArgumentParser(description='This tool will generate job description files for slurm and start the simulations. \n'
                                                 '')
    # Define the expected arguments
    parser.add_argument('platform_file', type=str, help='Path to the platform file')
    parser.add_argument('workload_file', type=str, help='Path to the workload file')
    parser.add_argument('dataset_file', type=str, help='Path to the dataset file')
    parser.add_argument('simulations_number', type=int, default=1, help='Number of simulations to run')
    parser.add_argument('max_simulation_duration_hours', type=int, default=10, help='Maximum duration of each simulation in hours')
    parser.add_argument('max_simulation_duration_minutes', type=int, default=0, help='Maximum duration of each simulation in minutes')

    # Parse the arguments
    args = parser.parse_args()

    # Check the arguments
    check_arguments(args, parser)

    # Check whether the files exist
    check_files_exist(args)

    # Check whether the dc-sim starter exist
    check_dcsim_exist()

    # Show simulation parameters
    simulation_hours_minutes = format_time(args.max_simulation_duration_hours, args.max_simulation_duration_minutes)
    print(f"Platform file: {args.platform_file}")
    print(f"Workload file: {args.workload_file}")
    print(f"Dataset file: {args.dataset_file}")
    print(f"Simulations number: {args.simulations_number}")
    print(f"Max simulation duration: {simulation_hours_minutes}")
    print(SEPARATOR)

    # Generate simulation id's with current date and uuid
    simulation_uuids = generate_uuids(args.simulations_number)

    prepare_simulations(simulation_uuids)
    check_integrity(simulation_uuids)
    start_simulations(simulation_uuids)


if __name__ == "__main__":
    main()
