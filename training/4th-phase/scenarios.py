import pandas as pd

TRAIN_DATASET_FILE = "train_dataset.csv"
TEST_DATASET_FILE = "test_dataset.csv"
NODES_FILE = "nodes_aux.csv"
LINKS_FILE = "links_aux.csv"

PATH_PREFIX = "../../dataset_preparation/4th-phase"



SCENARIOS_NAMES = [
    "0-main-dataset",
    '1-sgbatch-normal-simple-jobs',
    '2-sgbatch-high-links-latency-simple-jobs',
    '3-sgbatch-low-link-bandwidth-medium-latency-complex-jobs',
    '4-sgbatch-more-cores-complex-jobs',
    '5-sgbatch-less-cores-more-ram',
    '6-sgbatch-slower-storage-disk',
    '7-sgbatch-one-host-less',
    '8-kit_desy-normal',
    '9-kit_desy-no-fatpipes',
    '10-kit_desy-high-link-bandwidth',
    '11-kit_desy-less-cores-everywhere',
    '12-kit_desy-cache-moved-to-desy',
    '13-kit_desy-cache-moved-to-desy-faster-cores-mf-everywhere',
    '14-kit_desy-cache-moved-to-desy-less-cores-and-ram-in-kit',
    '15-kit_desy-cache-moved-to-desy-more-cores-in-desy']

input_columns_jobs_numerical = ['simulation_length', 'index', 'submission_time', 'flops', 'input_files_size', 'output_files_size']
input_columns_jobs_categorical = ['dataset_node_index']

output_columns_jobs_numerical = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']
output_columns_jobs_categorical = ['machine_index']

nodes_columns_numerical = ['speed_mf', 'cores', 'ram_gib', 'disk_tib', 'disk_read_bw_mbps', 'disk_write_bw_mbps']
nodes_columns_categorical = ['index', 'type_index', 'in_cluster']

links_columns_numerical = ['bandwidth_mbps', 'latency_us']
links_columns_categorical = ['link_index', 'src_node_index', 'dst_node_index', 'is_fatpipe']


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

main_scenario = scenarios[0]