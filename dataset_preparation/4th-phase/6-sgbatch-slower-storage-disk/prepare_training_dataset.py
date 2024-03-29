import sys
import os

import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prepare_dataset_commons as pdc

# Configuration
DATASET_LOCATION_INDEX = 5
NODES_FILE = 'nodes_aux.csv'
SUBMISSION_TIME = 0

PLATFORM_FILE_NAME = 'sgbatch-slower-storage-disk.xml'
WORKLOAD_FILE_NAME_PREFIX = 'T1_DE_KIT_workloads_'
DATASET_FILE_NAME_PREFIX = 'realData_RemoteStorage'

SIMULATIONS_OF_EACH_LENGTH_TRAINING = 7
SIMULATIONS_OF_EACH_LENGTH_TEST = 3

OUTPUT_TRAINING_FILE = "train_dataset.csv"
OUTPUT_TEST_FILE = "test_dataset.csv"

# Retrieve train and test dataset
train_df_original = pdc.get_dataset(platform=PLATFORM_FILE_NAME, workload_prefix=WORKLOAD_FILE_NAME_PREFIX, dataset_prefix=DATASET_FILE_NAME_PREFIX, simulations_of_each_length=SIMULATIONS_OF_EACH_LENGTH_TRAINING)
train_simulations = pdc.get_all_unique_simulation_ids(train_df_original)
test_df_original = pdc.get_dataset(platform=PLATFORM_FILE_NAME, workload_prefix=WORKLOAD_FILE_NAME_PREFIX, dataset_prefix=DATASET_FILE_NAME_PREFIX, simulations_of_each_length=SIMULATIONS_OF_EACH_LENGTH_TEST, exclude_simulations=train_simulations)

# Add column with dataset location node
train_df_with_dataset_location = pdc.add_dataset_node_index(train_df_original, node_index=DATASET_LOCATION_INDEX)
test_df_with_dataset_location = pdc.add_dataset_node_index(test_df_original, node_index=DATASET_LOCATION_INDEX)

# Add column with submission time
train_df_with_submission_time = pdc.add_submission_time(train_df_original, submission_time=SUBMISSION_TIME)
test_df_with_submission_time = pdc.add_submission_time(test_df_original, submission_time=SUBMISSION_TIME)

# Add column with node index
nodes_df = pd.read_csv(NODES_FILE, sep=";")
train_df_with_node_index = pdc.add_job_node_index(train_df_with_submission_time, nodes_df=nodes_df)
test_df_with_node_index = pdc.add_job_node_index(test_df_with_submission_time, nodes_df=nodes_df)

# Save files
train_df_with_node_index.to_csv(OUTPUT_TRAINING_FILE, index=False, sep=";")
test_df_with_node_index.to_csv(OUTPUT_TEST_FILE, index=False, sep=";")
