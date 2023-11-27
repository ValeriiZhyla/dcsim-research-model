import argparse
import os
import datetime
import random
import subprocess
import time
import uuid
import shutil

# Constants
DCSIM_EXECUTABLE_PATH = "~/dc-sim"
SEPARATOR = "============================================================"
BATCH_PREFIX = "simulation"
MAIL_FOR_ERRORS = "jn1292@partner.kit.edu"
BASH_SHEBANG = "#!/bin/bash"
SLURM_OUTPUT_FILE = "slurm-output.txt"
DCSIM_OUTPUT_FILE = "dcsim-output.txt"
DCSIM_SIMULATION_RESULT_FILE = "simulation-result.csv"
SLURM_JOB_FILE = "slurm-job.sh"

# Global variables
simulations_batch = None


def check_arguments(args, parser):
    if not (args.max_simulation_duration_hours >= 1):
        parser.error("max_simulation_duration_hours must be >= 1")
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
    full_dcsim_path = os.path.expanduser(DCSIM_EXECUTABLE_PATH)
    if not (os.path.isfile(full_dcsim_path) or os.path.islink(full_dcsim_path)):
        raise FileNotFoundError(f"The DCSim executable does not exist: {full_dcsim_path}")


def format_time(hours):
    return f"{hours:02}:00:00"


def generate_simulation_ids(n):
    uuids = []
    for _ in range(n):
        # Get the current date
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        # Generate a unique identifier
        unique_id = f"{date_str}-{uuid.uuid4()}"
        uuids.append(unique_id)
    return uuids


def create_simulations_batch_root_directory(batch_id: str):
    # Path for the new directory
    home_path = os.path.expanduser("~")
    full_path = os.path.join(home_path, batch_id)
    # Create root directory
    try:
        os.makedirs(full_path)
        simulation_root_directory = full_path
        return simulation_root_directory
    except Exception as e:
        raise IOError(f"Error creating simulation batch root directory: {e}")


def generate_batch_id() -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    id = f"{BATCH_PREFIX}_{timestamp}"
    return id


def run_squeue():
    print("=> squeue")
    subprocess.run(["squeue"], check=True)


class Simulation:
    id: str = ""
    platform_file_path: str = ""
    workload_file_path: str = ""
    dataset_file_path: str = ""
    duration_hours_minutes: str = ""
    directory_path: str = ""
    seed: int = 0

    SEED_MIN: int = 1
    SEED_MAX: int = 1000000000

    def __init__(self, id: str, platform_file_path: str, workload_file_path: str, dataset_file_path: str, duration_hours_minutes_seconds, root_dir: str):
        self.id = id
        self.platform_file_path = platform_file_path
        self.workload_file_path = workload_file_path
        self.dataset_file_path = dataset_file_path
        self.duration_hours_minutes_seconds = duration_hours_minutes_seconds
        self.directory_path = os.path.join(root_dir, id)
        self.seed = self.generate_seed_from_id(id)

    def generate_seed_from_id(self, id):
        # Use the hash of the string as the seed
        seed = hash(id)
        random.seed(seed)
        # Generate a random number between 1 and 1.000.000.000
        return random.randint(self.SEED_MIN, self.SEED_MAX)

    def create_directory(self):
        try:
            os.makedirs(self.directory_path)
        except Exception as e:
            raise IOError(f"Error creating simulation directory: {e}")

    def move_files(self):
        shutil.copy(self.platform_file_path, self.directory_path)
        shutil.copy(self.workload_file_path, self.directory_path)
        shutil.copy(self.dataset_file_path, self.directory_path)

    def create_slurm_job_file(self):
        slurm_file_content = [BASH_SHEBANG,
                              f"#SBATCH --job-name={self.id}",
                              f"#SBATCH --output={self.directory_path}/{SLURM_OUTPUT_FILE}",
                              f"#SBATCH --error={self.directory_path}/{DCSIM_OUTPUT_FILE}",
                              f"",
                              f"#SBATCH --ntasks=1",
                              f"#SBATCH --time={self.duration_hours_minutes_seconds}",
                              f"#SBATCH --partition=single",
                              f"",
                              f"#SBATCH --mail-user={MAIL_FOR_ERRORS}",
                              f"#SBATCH --mail-type=FAIL",
                              f"",
                              f"#PLATFORM: {self.platform_file_path}",
                              f"#WORKLOAD: {self.workload_file_path}",
                              f"#DATASET: {self.dataset_file_path}",
                              f"#SEED: {self.seed}",
                              f"",
                              f"# Execute a command",
                              f"{DCSIM_EXECUTABLE_PATH} --platform {self.platform_file_path} --workload-configurations {self.workload_file_path} --dataset-configurations {self.dataset_file_path} --output-file {self.directory_path}/{DCSIM_SIMULATION_RESULT_FILE} --seed {self.seed}"
                              ]
        slurm_file_path = os.path.join(self.directory_path, SLURM_JOB_FILE)
        with open(slurm_file_path, 'w') as file:
            for line in slurm_file_content:
                # Write each string to the file
                file.write(line + '\n')

    def start(self):
        slurm_file_path = os.path.join(self.directory_path, "slurm-job.sh")
        start_slurm_job_cmd = ["sbatch", slurm_file_path]
        result = subprocess.run(start_slurm_job_cmd, capture_output=True, text=True, check=True)
        print(f"Simulation started [{slurm_file_path}]: {result.stdout.strip()}")


class SimulationBatch:
    id: str = ""
    simulations: list[Simulation] = []
    root_path: str = ""

    def __init__(self, id: str, simulations: list[Simulation], root_path: str):
        self.id = id
        self.simulations = simulations
        self.root_path = root_path

    def prepare_simulations(self):
        for simulation in self.simulations:
            simulation.create_directory()
            simulation.move_files()
            simulation.create_slurm_job_file()
        print("=> Simulation directories prepared")
        print(SEPARATOR)

    def start_simulations(self):
        for simulation in self.simulations:
            simulation.start()
        print(f"=> All simulations started [{len(self.simulations)}]")
        print(SEPARATOR)


def main():
    parser = argparse.ArgumentParser(description='This tool generates job description files for slurm and start the simulations.\n'
                                                 'A directory is be prepared for each DCSim simulation, all input files, slurm job input and result file will be saved in this directory.\n'
                                                 'Each simulation has own uuid, which is used as a directory name. Simulation batch also has an uuid.\n'
                                                 f'Simulation result of DCSim is saved in [{DCSIM_SIMULATION_RESULT_FILE}].\n'
                                                 f'Text output of DCSim is saved in [{DCSIM_OUTPUT_FILE}].\n'
                                                 f'Slurm job description is saved in [{SLURM_JOB_FILE}].\n'
                                                 f'Slurm job result is saved in [{SLURM_OUTPUT_FILE}].\n',
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Define the expected arguments
    parser.add_argument('--simulations_number', type=int, default=1, help='Number of simulations to run', required=True)
    parser.add_argument('--max_simulation_duration_hours', type=int, default=10, help='Maximum duration of each simulation in hours', required=True)
    parser.add_argument('--platform_file', type=str, help='Path to the platform file', required=True)
    parser.add_argument('--workload_file', type=str, help='Path to the workload file', required=True)
    parser.add_argument('--dataset_file', type=str, help='Path to the dataset file', required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Check the arguments
    check_arguments(args, parser)

    # Check whether the files exist
    check_files_exist(args)

    # Check whether the dc-sim starter exist
    check_dcsim_exist()

    # Show simulation parameters
    simulation_hours_minutes_seconds = format_time(args.max_simulation_duration_hours)
    print(SEPARATOR)
    print(f"Platform file: {args.platform_file}")
    print(f"Workload file: {args.workload_file}")
    print(f"Dataset file: {args.dataset_file}")
    print(f"Simulations number: {args.simulations_number}")
    print(f"Max simulation duration: {simulation_hours_minutes_seconds}")
    print(SEPARATOR)

    # Create directory for simulation batch
    batch_id: str = generate_batch_id()
    batch_root_directory = create_simulations_batch_root_directory(batch_id)
    print(f"Simulation root directory: [{batch_root_directory}]")

    # Generate simulation id's with current date and uuid
    simulation_uuids = generate_simulation_ids(args.simulations_number)
    # create simulation objects
    simulations: list[Simulation] = [Simulation(id, args.platform_file, args.workload_file, args.dataset_file, simulation_hours_minutes_seconds, batch_root_directory) for id in
                                     simulation_uuids]
    # create simulation batch object
    simulation_batch: SimulationBatch = SimulationBatch(batch_id, simulations, batch_root_directory)

    # prepare and start simulations
    simulation_batch.prepare_simulations()
    time.sleep(1)
    simulation_batch.start_simulations()

    # show simulation jobs
    run_squeue()


if __name__ == "__main__":
    main()
