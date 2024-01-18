import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from initial_training.transformer_initial_training_batch_normalisation import TransformerModelWithTwoAuxEncodersAndNormalisation, CombinedDataset, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE

import windowing

# Constants
NUM_EPOCHS_ON_EACH_SCENARIO = 100
NUM_ITERATIONS_OVER_ALL_SCENARIOS = 10

TRAIN_DATASET_FILE = "train_dataset.csv"
TEST_DATASET_FILE = "test_dataset.csv"
NODES_FILE = "nodes_aux.csv"
LINKS_FILE = "links_aux.csv"

PATH_PREFIX = "../../../dataset-preparation/4th-phase"
SCENARIOS_NAMES = ['1-sgbatch-normal-simple-jobs', '2-sgbatch-high-links-latency-simple-jobs', '3-sgbatch-low-link-bandwidth-medium-latency-complex-jobs',
                   '4-sgbatch-more-cores-complex-jobs', '5-sgbatch-less-cores-more-ram', '6-sgbatch-slower-storage-disk', '7-sgbatch-one-host-less', '8-kit_desy-normal',
                   '9-kit_desy-no-fatpipes', '10-kit_desy-high-link-bandwidth', '11-kit_desy-less-cores-everywhere', '12-kit_desy-cache-moved-to-desy',
                   '13-kit_desy-cache-moved-to-desy-faster-cores-mf-everywhere', '14-kit_desy-cache-moved-to-desy-less-cores-and-ram-in-kit',
                   '15-kit_desy-cache-moved-to-desy-more-cores-in-desy']

input_columns_jobs = ['index', 'flops', 'input_files_size', 'output_files_size', 'dataset_node_index']
output_columns_jobs = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time', 'machine_index']
nodes_columns = ['index', 'type_index', 'speed_mf', 'cores', 'ram_gib', 'disk_tib', 'disk_read_bw_mbps', 'disk_write_bw_mbps', 'in_cluster']
links_columns = ['link_index', 'src_node_index', 'dst_node_index', 'bandwidth_mbps', 'latency_us', 'is_fatpipe']


class Scenario:
    name = ""
    train_dataset_path = ""
    test_dataset_path = ""
    nodes_aux_path = ""
    links_aux_path = ""

    def __init__(self, name, train_dataset_path, test_dataset_path, nodes_aux_path, links_aux_path):
        self.name = name
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.nodes_aux_path = nodes_aux_path
        self.links_aux_path = links_aux_path

    def read_train_dataset(self):
        return pd.read_csv(self.train_dataset_path, sep=";")

    def read_test_dataset(self):
        return pd.read_csv(self.test_dataset_path, sep=";")

    def read_nodes_aux(self):
        return pd.read_csv(self.nodes_aux_path, sep=";")

    def read_links_aux(self):
        return pd.read_csv(self.links_aux_path, sep=";")

    def read_all_datasets(self):
        return self.read_train_dataset(), self.read_test_dataset(), self.read_nodes_aux(), self.read_links_aux()


scenarios: list[Scenario] = list(map(lambda name: Scenario(
    name,
    f"{PATH_PREFIX}/{name}/{TRAIN_DATASET_FILE}",
    f"{PATH_PREFIX}/{name}/{TEST_DATASET_FILE}",
    f"{PATH_PREFIX}/{name}/{NODES_FILE}",
    f"{PATH_PREFIX}/{name}/{LINKS_FILE}"), SCENARIOS_NAMES))


def fine_tune_model(model: TransformerModelWithTwoAuxEncodersAndNormalisation):
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    for _ in range(0, NUM_ITERATIONS_OVER_ALL_SCENARIOS):
        for scenario in scenarios:
            # Load dataframes
            train_df, test_df, nodes_aux_df, links_aux_df = scenario.read_all_datasets()

            # Load jobs
            train_windows = windowing.create_windows_jobs(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns_jobs,
                                                          output_columns=output_columns_jobs)

            # Load aux data
            nodes_aux_df = nodes_aux_df[nodes_columns]
            links_aux_df = links_aux_df[links_columns]
            nodes_aux_windows = windowing.pad_window_aux(nodes_aux_df, window_size=WINDOW_SIZE, columns=nodes_columns)
            links_aux_windows = windowing.pad_window_aux(links_aux_df, window_size=WINDOW_SIZE, columns=links_columns)

            # Fit the scalers on the whole training dataset
            input_jobs_scaler = windowing.create_and_fit_scaler(train_df, input_columns_jobs)
            output_jobs_scaler = windowing.create_and_fit_scaler(train_df, output_columns_jobs)
            nodes_scaler = windowing.create_and_fit_scaler(nodes_aux_df, nodes_columns)
            links_scaler = windowing.create_and_fit_scaler(links_aux_df, links_columns)

            # Prepare datasets
            train_dataset = windowing.scale_and_reshape_jobs_windows(train_windows, WINDOW_SIZE, input_jobs_scaler, output_jobs_scaler, input_columns_jobs, output_columns_jobs)
            nodes_aux_dataset = windowing.scale_and_reshape_aux_windows(nodes_aux_windows, WINDOW_SIZE, nodes_scaler, nodes_columns)
            links_aux_dataset = windowing.scale_and_reshape_aux_windows(links_aux_windows, WINDOW_SIZE, links_scaler, links_columns)

            train_dataset_jobs_with_aux = CombinedDataset(train_dataset, nodes_aux_dataset, links_aux_dataset)

            train_loader = DataLoader(train_dataset_jobs_with_aux, batch_size=BATCH_SIZE, shuffle=True)

            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)  # smaller learning rate

            # Training Loop
            model.train()  # Set the model to training mode
            last_avg_loss = 0
            for epoch in range(NUM_EPOCHS_ON_EACH_SCENARIO):
                total_loss = 0
                for batch in train_loader:
                    # Unpack the batch data
                    train_data, nodes_aux_data, links_aux_data, train_target = batch

                    # Move data to the device
                    train_data = train_data.to(device)
                    nodes_aux_data = nodes_aux_data.to(device)
                    links_aux_data = links_aux_data.to(device)
                    train_target = train_target.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(train_data, nodes_aux_data, links_aux_data, train_target)
                    loss = criterion(outputs, train_target)

                    # Backward pass
                    loss.backward()
                    # It helps, nan otherwise
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()

                    # Accumulate loss
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                last_avg_loss = avg_loss
                print(f'{scenario.name}: Epoch [{epoch + 1}/{NUM_EPOCHS_ON_EACH_SCENARIO}], Average Loss: {avg_loss}')

            # Print training summary
            print("=================================")
            print(f"Scenario: {scenario.name}")
            print(f"Last loss: {last_avg_loss}")
            print("=================================")

    end_time = time.time()
    # Stop timer
    total_time = end_time - start_time
    print(f"Total time for training: {total_time:.2f} seconds")

    return model


if __name__ == '__main__':
    model = torch.load('../../../trained-models/4th-phase/initial_model_batch_normalisation/transformer_with_batch_normalisation.pth')

    model = fine_tune_model(model)
    torch.save(model.state_dict(), '../generated-models/transformer_tuned_on_15_scenarios_weights.pth')
    torch.save(model, '../generated-models/transformer_tuned_on_15_scenarios.pth')
