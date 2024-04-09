from dataset_preparation import prepare_dataset_commons as pdc

# Configuration

PLATFORM_FILE_NAME = 'WLCG_disklessTier2_reduced100.xml'
WORKLOAD_FILE_NAME_PREFIX = 'T2_DE_DESY_workloads_'
DATASET_FILE_NAME_PREFIX = 'realData_GridKA'

SIMULATIONS_OF_EACH_LENGTH_TRAINING = 40
SIMULATIONS_OF_EACH_LENGTH_TEST = 50

OUTPUT_TRAINING_FILE = "train_dataset.csv"
OUTPUT_TEST_FILE = "interpolation_dataset.csv"

test_simulations = ['2620682f-34fa-43d1-b133-e51c6c2b20aa', '036424c6-2ba8-4be9-abc1-8c929e11b50e', 'b4db2a1f-592b-4649-ba2c-bf5b8a84141b', '1cf61b87-919e-45c9-a26e-a6e46dc0562c', 'ed9d1eae-fee4-4d63-8fef-d19a7009bb9d', '1e773381-71f3-4074-a253-d60f830d2b3a', '733baf96-c762-40fe-8860-eb7aebe5efbd', '97fd6264-d9eb-4323-bdc0-8bc5f10cc413', '8bfc232c-05fb-4ed5-8038-c0276546ae82', 'a241ec03-bfc8-4ae1-9482-a6320e6fa88e', 'f11705a1-0646-4528-b856-3b917076ace3', '75b39e25-0783-481a-aadb-123dcb96b598', 'fb8ceec3-b193-4073-a61a-ef219f845082', '3cf2239a-ce8d-466f-ab6a-b0beb439de0c', '27d1cdd3-cbfb-4de3-b276-6271ba6b44a9', 'ec279266-5a28-476f-b313-7fbbdc027798', '60060dc0-a1b3-4a9f-96a2-e56155723fcc', '7e674773-7803-4a7b-a746-1cff058204b4', 'f8bb9b31-17f5-4e0e-8d65-de263633a22a', '94c39012-ee17-482b-94eb-77c36d360ed2', 'ef3d6b54-1106-416a-92ff-91b92d141a94', 'fa27623b-c6e4-4279-a7db-6379eaab2897', '9b5d3eeb-49a2-4b5b-aaf5-0ae2f31911ea', '7ad26381-4386-4454-8c06-5b0162c1f756', 'dc41ab19-409a-478a-a24b-62a22930c228', '46ef1e4e-29e2-46a3-ad55-c96c0f67b356', '237f4d86-2ff2-4b53-b721-f1adfb6ecda5', '494e3482-6a1a-4a89-9e3b-b5368dde37e9', '8668757d-9d5b-41d7-8b7e-3d094e98cb05', 'aab53ec5-b4d8-4a01-b21f-c31353dc5820', '12ae7a02-7ce7-4284-800b-50509a9d16a0', '86b8bd26-9f37-45e2-afaa-cce1e50c4709', '4794fb2f-2f31-4886-b698-8310a22f786c', 'ca4e7277-d122-4d4e-9eed-0b9504296d06', '40f489af-b679-495c-9044-b9a35a162dc3', '80b46c27-dee8-4be1-b6f2-4cd1b32acddb', '2ff231bd-d192-4719-b53e-ffb3e8732dd4', '5df614b6-e322-46a7-b516-6ba23ec9222b', 'ee6a173f-5917-4286-bd9f-4a38de09d13b', 'b2602154-922f-4ff4-940e-723b8c3cb970', '4522ba6f-2060-43f1-9a00-a0b003f22980', '7ec51caa-6145-48fe-90c5-a33186df99eb', 'ba518c40-d9eb-4858-b5c4-b21822b68b44', 'f1670868-8d2d-4594-8ad2-8bb5d86fadc1', '5eb9c37a-f375-4308-a4bb-fbb024d387ab', '641a0eb1-7d5f-4db5-a644-b1d547584166', '15161d9e-3695-4982-83e8-251775c4b7c3', '0cd4b1a6-7ef7-4ec2-8165-5bf7d0786336', '3e144faf-623c-458d-8349-c884a4491224', '10a15b69-e9be-49cd-90e5-40748da321b6']


# Retrieve train and test dataset
train_df_original = pdc.get_dataset(platform=PLATFORM_FILE_NAME, workload_prefix=WORKLOAD_FILE_NAME_PREFIX, dataset_prefix=DATASET_FILE_NAME_PREFIX,
                                    simulations_of_each_length=SIMULATIONS_OF_EACH_LENGTH_TRAINING, table_simulated_jobs="third_phase_simulated_jobs_extrapolation_x5",
                                    exclude_simulations=test_simulations)
test_df_original = pdc.get_simulations_by_id(test_simulations, table_simulated_jobs="third_phase_simulated_jobs_extrapolation_x5")

# Add simulation id unique integer
train_df_with_unique_simulation_id_integers = pdc.add_simulation_id_integer(train_df_original)
test_df_with_unique_simulation_id_integers = pdc.add_simulation_id_integer(test_df_original)

# Save files
train_df_original.to_csv(OUTPUT_TRAINING_FILE, index=False, sep=";")
test_df_original.to_csv(OUTPUT_TEST_FILE, index=False, sep=";")
