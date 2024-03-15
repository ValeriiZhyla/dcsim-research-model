import sys
import os

import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prepare_dataset_commons as pdc

# Configuration
DATASET_LOCATION_INDEX = 17
NODES_FILE = 'nodes_aux.csv'
SUBMISSION_TIME = 10

PLATFORM_FILE_NAME = 'WLCG_disklessTier2_reduced100-storage-cache-switched-more-resources-in-desy-less-resources-in-kit-faster-links.xml'
WORKLOAD_FILE_NAME_PREFIX = 'T2_DE_DESY_workloads_'
DATASET_FILE_NAME_PREFIX = 'realData_DESY_cache-giant_10000'

# SIMULATIONS_OF_EACH_LENGTH_TRAINING = 7
# SIMULATIONS_OF_EACH_LENGTH_TEST = 3
SIMULATIONS_OF_EACH_LENGTH_EXTRAPOLATION = 50

OUTPUT_TRAINING_FILE = "train_dataset.csv"
OUTPUT_TEST_FILE = "test_dataset.csv"
OUTPUT_EXTRAPOLATION_FILE = "extrapolation_dataset.csv"

# Retrieve train and test dataset
extrapolation_df_original = pdc.get_dataset(platform=PLATFORM_FILE_NAME, workload_prefix=WORKLOAD_FILE_NAME_PREFIX, dataset_prefix=DATASET_FILE_NAME_PREFIX, simulations_of_each_length=SIMULATIONS_OF_EACH_LENGTH_EXTRAPOLATION, table_simulated_jobs="simulated_jobs_extrapolation")

# Add column with dataset location node
extrapolation_df_with_dataset_location = pdc.add_dataset_node_index(extrapolation_df_original, node_index=DATASET_LOCATION_INDEX)

# Add column with submission time
extrapolation_df_with_submission_time = pdc.add_submission_time(extrapolation_df_with_dataset_location, submission_time=SUBMISSION_TIME)


# Add column with node index
nodes_df = pd.read_csv(NODES_FILE, sep=";")
extrapolation_df_with_node_index = pdc.add_job_node_index(extrapolation_df_with_submission_time, nodes_df=nodes_df)

# Save files
extrapolation_df_with_node_index.to_csv(OUTPUT_EXTRAPOLATION_FILE, index=False, sep=";")