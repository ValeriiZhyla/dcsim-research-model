import argparse

import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
DB_NAME = "simulations"
DB_USER = "orator"
DB_PASSWORD = "orator"
DB_HOST = "localhost"
DB_PORT = "5432"

TEST_DATASET_SIZE = 0.3
SEED = 42


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


def get_all_unique_simulation_ids(train_df_original):
    return train_df_original['simulation_id'].unique()


def add_dataset_node_index(df, node_index):
    df['dataset_node_index'] = node_index
    return df


def add_job_node_index(df, nodes_df):
    machine_dict = dict(zip(nodes_df['name'], nodes_df['index']))
    # Trim spaces from machine_name values in df (bug in exporter for old data)
    df['machine_name'] = df['machine_name'].str.strip()
    df['machine_index'] = df['machine_name'].map(machine_dict)
    return df


def get_dataset(platform, workload_prefix, dataset_prefix, simulations_of_each_length, exclude_simulations=None):
    if exclude_simulations is None:
        exclude_simulations = []

    # Establish the database connection
    conn = establish_database_connection()

    # Read dataset
    query = (
        f"SELECT simulation_id, position_in_batch, tag, machine_name, hit_rate, job_start, job_end, compute_time, flops, input_files_transfer_time, input_files_size, output_files_transfer_time, output_files_size "
        f"FROM second_phase_simulated_jobs "
        f"JOIN simulations ON second_phase_simulated_jobs.simulation_id = simulations.id "
        f"WHERE simulations.platform_config_name = '{platform}' "
        f"AND simulations.dataset_config_name LIKE '{dataset_prefix}%' "
        f"AND simulations.workload_config_name LIKE '{workload_prefix}%';")
    print("Query:", query)

    df_all = pd.read_sql_query(query, conn)

    # Exclude rows where simulation_id is in the exclude_simulations list
    df_all = df_all[~df_all['simulation_id'].isin(exclude_simulations)]

    # Sort entries by id and start time
    df_sorted = df_all.sort_values(by=['simulation_id', 'job_start'])

    # Add an index within each simulation group
    df_sorted['index'] = df_sorted.groupby('simulation_id').cumcount() + 1

    # Count the number of simulations for each simulation_id
    df_sorted['simulation_length'] = df_sorted.groupby('simulation_id')['simulation_id'].transform('count')

    # Select a specific number of simulations of each length randomly
    df_selected_simulations = df_sorted.groupby('simulation_length').sample(n=simulations_of_each_length, replace=False)
    # Print simulation_id and its corresponding length
    for length, group in df_selected_simulations.groupby('simulation_length'):
        print(f"Simulation Length {length} simulations: {group['simulation_id'].unique()}")

    unique_simulation_ids = df_selected_simulations['simulation_id'].unique()

    df_sorted_filtered = df_sorted[df_sorted['simulation_id'].isin(unique_simulation_ids)]
    # Close the database connection
    conn.close()

    return df_sorted_filtered
