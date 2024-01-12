import pandas as pd


def add_columns_to_csv(input_csv_file, output_csv_file, nodes_file):
    # Read the CSV file
    df = pd.read_csv(input_csv_file)
    nodes_df = pd.read_csv(nodes_file)

    machine_dict = dict(zip(nodes_df['name'], nodes_df['index']))

    # Add a new column 'dataset_node_index' with all values set to 13
    df['dataset_node_index'] = 13
    df['machine_index'] = df['machine_name'].map(machine_dict)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv_file, index=False)


# File paths
input_csv_file = '../../dataset-preparation/4th-phase/main-dataset/train_dataset_small.csv'
output_csv_file = '../../dataset-preparation/4th-phase/main-dataset/train_dataset_small_extended.csv'
nodes_file = '../../dataset-preparation/4th-phase/main-dataset/nodes_aux.csv'
add_columns_to_csv(input_csv_file, output_csv_file, nodes_file)

input_csv_file = '../../dataset-preparation/4th-phase/main-dataset/test_dataset_small.csv'
output_csv_file = '../../dataset-preparation/4th-phase/main-dataset/test_dataset_small_extended.csv'
add_columns_to_csv(input_csv_file, output_csv_file, nodes_file)

input_csv_file = '../../dataset-preparation/4th-phase/main-dataset/train_dataset.csv'
output_csv_file = '../../dataset-preparation/4th-phase/main-dataset/train_dataset_extended.csv'
add_columns_to_csv(input_csv_file, output_csv_file, nodes_file)

input_csv_file = '../../dataset-preparation/4th-phase/main-dataset/test_dataset.csv'
output_csv_file = '../../dataset-preparation/4th-phase/main-dataset/test_dataset_extended.csv'
add_columns_to_csv(input_csv_file, output_csv_file, nodes_file)

input_csv_file = '../../dataset-preparation/4th-phase/main-dataset/test_dataset_small.csv'
output_csv_file = '../../dataset-preparation/4th-phase/main-dataset/test_dataset_small_extended.csv'
add_columns_to_csv(input_csv_file, output_csv_file, nodes_file)

input_csv_file = '../../dataset-preparation/4th-phase/main-dataset/extrapolation_dataset_x20.csv'
output_csv_file = '../../dataset-preparation/4th-phase/main-dataset/extrapolation_dataset_x20_extended.csv'
add_columns_to_csv(input_csv_file, output_csv_file, nodes_file)
