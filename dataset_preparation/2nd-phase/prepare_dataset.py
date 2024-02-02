from dataset_preparation import prepare_dataset_commons as pdc

# Configuration

PLATFORM_FILE_NAME = 'sgbatch.xml'
WORKLOAD_FILE_NAME_PREFIX = 'T1_DE_KIT_workloads_'
DATASET_FILE_NAME_PREFIX = 'realData_RemoteStorage'

SIMULATIONS_OF_EACH_LENGTH_TRAINING = 700
SIMULATIONS_OF_EACH_LENGTH_TEST = 300
SIMULATIONS_OF_EACH_LENGTH_EXTRAPOLATION = 10

OUTPUT_TRAINING_FILE = "train_dataset.csv"
OUTPUT_TEST_FILE = "test_dataset.csv"
OUTPUT_EXTRAPOLATION_FILE = "extrapolation_dataset.csv"

# Retrieve train and test dataset
train_df_original = pdc.get_dataset(platform=PLATFORM_FILE_NAME, workload_prefix=WORKLOAD_FILE_NAME_PREFIX, dataset_prefix=DATASET_FILE_NAME_PREFIX,
                                    simulations_of_each_length=SIMULATIONS_OF_EACH_LENGTH_TRAINING, table_simulated_jobs="second_phase_simulated_jobs")
train_simulations_ids = pdc.get_all_unique_simulation_ids(train_df_original)
test_df_original = pdc.get_dataset(platform=PLATFORM_FILE_NAME, workload_prefix=WORKLOAD_FILE_NAME_PREFIX, dataset_prefix=DATASET_FILE_NAME_PREFIX,
                                   simulations_of_each_length=SIMULATIONS_OF_EACH_LENGTH_TEST, exclude_simulations=train_simulations_ids,
                                   table_simulated_jobs="second_phase_simulated_jobs")
extrapolation_df_original = pdc.get_dataset(platform=PLATFORM_FILE_NAME, workload_prefix=WORKLOAD_FILE_NAME_PREFIX, dataset_prefix=DATASET_FILE_NAME_PREFIX,
                                            simulations_of_each_length=SIMULATIONS_OF_EACH_LENGTH_EXTRAPOLATION, table_simulated_jobs="second_phase_simulated_jobs_extrapolation")

# Add simulation id unique integer
train_df_with_unique_simulation_id_integers = pdc.add_simulation_id_integer(train_df_original)
test_df_with_unique_simulation_id_integers = pdc.add_simulation_id_integer(test_df_original)
extrapolation_df_with_unique_simulation_id_integers = pdc.add_simulation_id_integer(extrapolation_df_original)

# Save files
train_df_original.to_csv(OUTPUT_TRAINING_FILE, index=False, sep=";")
test_df_original.to_csv(OUTPUT_TEST_FILE, index=False, sep=";")
extrapolation_df_original.to_csv(OUTPUT_EXTRAPOLATION_FILE, index=False, sep=";")
