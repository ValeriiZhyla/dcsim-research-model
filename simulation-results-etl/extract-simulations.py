import argparse
import csv
import os.path
import uuid
from datetime import datetime
from pathlib import Path

import psycopg2

# Configuration
DB_NAME = "simulations"
DB_USER = "orator"
DB_PASSWORD = "orator"
DB_HOST = "localhost"
DB_PORT = "5432"

# Constants
SEPARATOR = "============================================================"
DCSIM_OUTPUT_FILE = "dcsim-output.txt"
DCSIM_SIMULATION_RESULT_FILE = "simulation-result.csv"
# slurm job
SLURM_JOB_FILE = "slurm-job.sh"
SLURM_JOB_FILE_PLATFORM_PARAMETER = "#PLATFORM:"
SLURM_JOB_FILE_WORKLOAD_PARAMETER = "#WORKLOAD:"
SLURM_JOB_FILE_DATASET_PARAMETER = "#DATASET:"
SLURM_JOB_FILE_SEED_PARAMETER = "#SEED:"
# slurm output
SLURM_OUTPUT_FILE = "slurm-output.txt"
SLURM_OUTPUT_FILE_STATE_PARAMETER = "State:"
SLURM_OUTPUT_FILE_CPU_TIME_PARAMETER = "CPU Utilized:"
SLURM_OUTPUT_FILE_MEMORY_PARAMETER = "Memory Utilized:"

SIMULATION_RESULT_CSV_HEADER = "job.tag, machine.name, hitrate, job.start, job.end, job.computetime, job.flops, infiles.transfertime, infiles.size, outfiles.transfertime, outfiles.size"
EXPECTED_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

COMPLETED_STATE = "COMPLETED"

EXTRAPOLATION = ["extrapolation", "ext"]

SIMULATED_JOBS_EXTRAPOLATION_TABLE = "simulated_jobs_extrapolation"
SIMULATED_JOBS_TABLE = "simulated_jobs"


def check_arguments(args, parser):
    if not args.simulation_root_dir.strip():
        parser.error("simulation_root_dir argument is empty")


def check_directory_exist(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")


def establish_database_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.Error as e:
        print("Error: Could not make the connection to the postgres database")
        raise e


def get_simulation_directories(simulation_root: str) -> list[str]:
    return [d.name for d in Path(simulation_root).iterdir() if d.is_dir()]


class SimulationDirectory:
    # from job id
    creation_timestamp: str = ""
    uuid: str = ""

    # from slurm-output.txt
    slurm_output_file_content: str = ""
    state: str = ""
    cpu_time: str = ""
    memory: str = ""

    # from simulation-result.csv
    simulation_result: str = ""

    # from slurm_job.sh
    slurm_job_file_content: str = ""
    platform_file_name: str = ""
    workload_file_name: str = ""
    dataset_file_name: str = ""
    simulation_seed: str = ""

    # files saved in directory, names in slurm-job.sh
    platform_file_content: str = ""
    workload_file_content: str = ""
    dataset_file_content: str = ""

    # Check flags
    is_initialized = False
    is_valid = False

    def __init__(self, creation_timestamp, uuid):
        self.creation_timestamp = creation_timestamp
        self.uuid = uuid

    def add_slurm_job_file_content(self, slurm_job_file_content):
        self.slurm_job_file_content = slurm_job_file_content

    def add_dcsim_filenames(self, platform_file_name, workload_file_name, dataset_file_name):
        self.platform_file_name = platform_file_name
        self.workload_file_name = workload_file_name
        self.dataset_file_name = dataset_file_name

    def add_dcsim_files_content(self, platform_file_content, workload_file_content, dataset_file_content):
        self.platform_file_content = platform_file_content
        self.workload_file_content = workload_file_content
        self.dataset_file_content = dataset_file_content

    def add_seed(self, seed):
        self.simulation_seed = seed

    def add_slurm_output_file_content(self, slurm_output_file_content):
        self.slurm_output_file_content = slurm_output_file_content

    def add_slurm_simulation_metadata(self, state, cpu_time, memory):
        self.state = state
        self.cpu_time = cpu_time
        self.memory = memory

    def add_simulation_result_file_content(self, simulation_result_file_content):
        self.simulation_result = simulation_result_file_content

    def check_all_parameters_are_set(self):
        if any(attr == "" for attr in vars(self).values()):
            raise AttributeError(f"{self.creation_timestamp}-{self.uuid}: At least one attribute is an empty string.")

    def check_simulation_result_header(self):
        lines = self.simulation_result.split("\n")
        header = lines[0]
        if not (header == SIMULATION_RESULT_CSV_HEADER):
            raise AttributeError(f"{self.creation_timestamp}-{self.uuid}: simulation_result header: [{header}]. Expected: [{SIMULATION_RESULT_CSV_HEADER}]")

    def check_timestamp(self):
        try:
            # Try to parse the string as a timestamp
            datetime.strptime(self.creation_timestamp, EXPECTED_TIMESTAMP_FORMAT)
        except ValueError:
            # String does not match the timestamp format
            raise AttributeError(
                f"{self.creation_timestamp}-{self.uuid}: creation_timestamp: [{self.creation_timestamp}] does not match expected format: [{EXPECTED_TIMESTAMP_FORMAT}]")

    def check_uuid(self):
        uuid_obj = uuid.UUID(self.uuid)
        if not str(uuid_obj) == self.uuid:
            raise AttributeError(f"{self.creation_timestamp}-{self.uuid}: uuid: [{self.uuid}] is not a valid uuid")

    def check_constraints(self):
        self.check_all_parameters_are_set()
        self.check_timestamp()
        self.check_uuid()
        self.check_simulation_result_header()

    def save_in_database_if_not_exist(self, conn, table_for_jobs: str):
        if not self.is_already_in_database(conn):
            self.save_in_database(conn, table_for_jobs)
        else:
            print(f"Simulation [{self.creation_timestamp}-{self.uuid}] is already in database")

    def is_already_in_database(self, conn):
        with conn.cursor() as cursor:
            query = f"SELECT 1 FROM simulations WHERE id = %s"
            cursor.execute(query, (self.uuid,))
            # Fetch the result
            result = cursor.fetchone()
            return result is not None

    def save_in_database(self, conn, table_name: str):
        try:
            with conn.cursor() as cursor:
                # transaction start
                self.fill_simulations_table(cursor)
                self.fill_slurm_metadata_table(cursor)
                self.fill_jobs_table(cursor, table_name)
                # transaction end
            conn.commit()
            print(f"Simulation [{self.creation_timestamp}-{self.uuid}] was saved ")
        except Exception as e:
            # Perform rollback on any error
            conn.rollback()
            print(f"Error during saving the simulation [{self.creation_timestamp}-{self.uuid}]: {e}")

    def fill_simulations_table(self, cursor):
        insert_simulation_query = """
                    INSERT INTO simulations (id, seed, platform_config, dataset_config, workload_config, platform_config_name, dataset_config_name, workload_config_name, created_at, simulations_result)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
        # Execute the INSERT statement for the simulations table
        cursor.execute(insert_simulation_query, (
            self.uuid,  # Convert string to UUID
            int(self.simulation_seed),  # Assuming seed is an integer
            self.platform_file_content,
            self.dataset_file_content,
            self.workload_file_content,
            self.platform_file_name,
            self.dataset_file_name,
            self.workload_file_name,
            datetime.strptime(self.creation_timestamp, "%Y%m%d-%H%M%S"),  # Convert string to timestamp
            self.simulation_result
        ))

    def fill_slurm_metadata_table(self, cursor):
        insert_slurm_metadata_query = """
                    INSERT INTO slurm_execution_metadata (simulation_id, cpu_time_text, memory_used_text, slurm_output, slurm_job_description)
                    VALUES (%s, %s, %s, %s, %s);
                """
        # Execute the INSERT statement for the slurm_execution_metadata table
        cursor.execute(insert_slurm_metadata_query, (
            self.uuid,  # Convert string to UUID
            self.cpu_time,
            self.memory,
            self.slurm_output_file_content,
            self.slurm_job_file_content
        ))

    def fill_jobs_table(self, cursor, table_name):
        insert_job_query = f"""
                    INSERT INTO {table_name} (simulation_id, position_in_batch, tag, machine_name, hit_rate, job_start, job_end, compute_time, flops, input_files_transfer_time, input_files_size, output_files_transfer_time, output_files_size)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
        # Parse the CSV data from simulation_result
        reader = csv.reader(self.simulation_result.splitlines())
        next(reader)  # Skip the header row
        for i, row in enumerate(reader):
            # Extract fields from the row
            tag, machine_name, hit_rate, job_start, job_end, compute_time, flops, infiles_transfertime, infiles_size, outfiles_transfertime, outfiles_size = row
            position_in_batch = i + 1  # position in batch, start with 1, i starts with 0
            # Execute the INSERT statement for each job
            cursor.execute(insert_job_query, (
                self.uuid,  # Convert string to UUID
                position_in_batch,  # position in batch, start with
                tag,
                machine_name,
                float(hit_rate),
                float(job_start),
                float(job_end),
                float(compute_time),
                float(flops),
                float(infiles_transfertime),
                float(infiles_size),
                float(outfiles_transfertime),
                float(outfiles_size)
            ))


def check_expected_files_present(full_dir_path):
    files_to_check = [SLURM_OUTPUT_FILE, DCSIM_OUTPUT_FILE, DCSIM_SIMULATION_RESULT_FILE, SLURM_JOB_FILE]
    for filename in files_to_check:
        file_path = os.path.join(full_dir_path, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            continue
        else:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Expected file {file_path} does not exist (unexpected state)")
            elif not os.path.getsize(file_path) > 0:
                raise FileNotFoundError(f"Expected file {file_path} exists but is empty (unexpected state)")


def extract_timestamp_and_uuid(dir_name) -> (str, str):
    # Split the directory name into two parts: timestamp and UUID
    parts = dir_name.split('-')
    timestamp = '-'.join(parts[:2])  # YYYYMMDD-HHMMSS
    uuid = '-'.join(parts[2:])  # UUID
    return timestamp, uuid


def read_file_content(directory, filename) -> str:
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    # Open the file and read its content
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def get_value_from_line_that_starts_with(file_content: str, parameter_line_start: str) -> str:
    lines = file_content.split("\n")
    for line in lines:
        if line.startswith(parameter_line_start):
            line = line.replace(parameter_line_start, "")
            line = line.strip()
            return line
    raise KeyError(f"No line starts with [{parameter_line_start}] in [{lines}]")


def create_directory_entity(simulation_root_dir, dir_name) -> SimulationDirectory:
    full_dir_path = os.path.join(simulation_root_dir, dir_name)
    check_expected_files_present(full_dir_path)
    timestamp, uuid = extract_timestamp_and_uuid(dir_name)
    # parse slurm_job.sh to get all simulation inputs
    slurm_job_file_content = read_file_content(full_dir_path, SLURM_JOB_FILE)
    platform_file_name = get_value_from_line_that_starts_with(slurm_job_file_content, SLURM_JOB_FILE_PLATFORM_PARAMETER)
    workload_file_name = get_value_from_line_that_starts_with(slurm_job_file_content, SLURM_JOB_FILE_WORKLOAD_PARAMETER)
    dataset_file_name = get_value_from_line_that_starts_with(slurm_job_file_content, SLURM_JOB_FILE_DATASET_PARAMETER)
    simulation_seed = get_value_from_line_that_starts_with(slurm_job_file_content, SLURM_JOB_FILE_SEED_PARAMETER)
    # read corresponding files
    platform_file_content = read_file_content(full_dir_path, platform_file_name)
    workload_file_content = read_file_content(full_dir_path, workload_file_name)
    dataset_file_content = read_file_content(full_dir_path, dataset_file_name)
    # parse slurm-output.txt
    slurm_output_file_content = read_file_content(full_dir_path, SLURM_OUTPUT_FILE)
    state = get_value_from_line_that_starts_with(slurm_output_file_content, SLURM_OUTPUT_FILE_STATE_PARAMETER)
    cpu_time = get_value_from_line_that_starts_with(slurm_output_file_content, SLURM_OUTPUT_FILE_CPU_TIME_PARAMETER)
    memory = get_value_from_line_that_starts_with(slurm_output_file_content, SLURM_OUTPUT_FILE_MEMORY_PARAMETER)

    # read simulation-result.csv
    simulation_results_file_content = read_file_content(full_dir_path, DCSIM_SIMULATION_RESULT_FILE)

    # build the object
    simulation_directory = SimulationDirectory(timestamp, uuid)
    simulation_directory.add_slurm_job_file_content(slurm_job_file_content)
    simulation_directory.add_dcsim_filenames(platform_file_name, workload_file_name, dataset_file_name)
    simulation_directory.add_dcsim_files_content(platform_file_content, workload_file_content, dataset_file_content)
    simulation_directory.add_seed(simulation_seed)
    simulation_directory.add_slurm_output_file_content(slurm_output_file_content)
    simulation_directory.add_slurm_simulation_metadata(state, cpu_time, memory)
    simulation_directory.add_simulation_result_file_content(simulation_results_file_content)
    simulation_directory.is_initialized = True

    # check the object constraints
    simulation_directory.check_constraints()
    simulation_directory.is_valid = True

    return simulation_directory


def filter_completed_simulations(simulation_directories: list[SimulationDirectory]) -> list[SimulationDirectory]:
    completed_simulations = []
    for simulation in simulation_directories:
        if COMPLETED_STATE in simulation.state:
            completed_simulations.append(simulation)
        else:
            print(
                f"Warning: simulation [{simulation.creation_timestamp}-{simulation.uuid}] is skipped. Cause: state is not [{COMPLETED_STATE}]. Simulation state: [{simulation.state}] ")
    return completed_simulations


def save_to_database_if_not_exist(conn, completed_simulations: list[SimulationDirectory], is_extrapolation_data: bool):
    if is_extrapolation_data:
        table_name = SIMULATED_JOBS_EXTRAPOLATION_TABLE
    else:
        table_name = SIMULATED_JOBS_TABLE
    print(f"Simulated jobs will be saved in table {table_name}")
    for simulation in completed_simulations:
        simulation.save_in_database_if_not_exist(conn, table_name)


def check_dir_contains_extrapolation(simulation_root_dir):
    return any(word in simulation_root_dir.lower() for word in EXTRAPOLATION)


def main():
    parser = argparse.ArgumentParser(
        description='This tool processes the directory with simulations and loads different parts of each simulation into corresponding database tables.',
        formatter_class=argparse.RawTextHelpFormatter)
    # Define the expected arguments
    parser.add_argument('--simulation_root_dir', type=str, required=True,
                        help='Root directory for simulations. Can be used to accumulate results from many runs. Default directory will be created, if no value provided.')

    # Parse the arguments
    args = parser.parse_args()

    # Check the arguments
    check_arguments(args, parser)

    # Check whether the directory exist
    simulation_root_dir = args.simulation_root_dir
    check_directory_exist(simulation_root_dir)
    is_extrapolation_data = check_dir_contains_extrapolation(simulation_root_dir)

    # Establish the database connection
    conn = establish_database_connection()

    simulation_directory_names = get_simulation_directories(simulation_root_dir)
    simulation_directories: list[SimulationDirectory] = [create_directory_entity(simulation_root_dir, name) for name in simulation_directory_names]
    completed_simulations: list[SimulationDirectory] = filter_completed_simulations(simulation_directories)
    save_to_database_if_not_exist(conn, completed_simulations, is_extrapolation_data)

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()
