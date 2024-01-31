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

# Constants
DEFAULT_TRAINING_FILE_PATH = "./train_dataset.csv"
DEFAULT_TEST_FILE_PATH = "./test_dataset.csv"
DEFAULT_SMALL_TRAINING_FILE_PATH = "./train_dataset_small.csv"
DEFAULT_SMALL_TEST_FILE_PATH = "./test_dataset_small.csv"

EXTRAPOLATION_X5_FILE_PATH = "./extrapolation_dataset_x5.csv"
EXTRAPOLATION_X10_FILE_PATH = "./extrapolation_dataset_x10.csv"
EXTRAPOLATION_X20_FILE_PATH = "./extrapolation_dataset_x20.csv"

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


def export_dataset_train_test(query, conn, training_file_path, testing_file_path):
    df = pd.read_sql_query(query, conn)

    # Sort the DataFrame by 'simulation_id' and then by 'job_start'
    # Need to sort by job start because it is not feasible to learn the behaviour of scheduler
    df_sorted = df.sort_values(by=['simulation_id', 'job_start'])

    # Add an index within each simulation group
    df_sorted['index'] = df_sorted.groupby('simulation_id').cumcount() + 1

    # Group by 'simulation_id' and split
    unique_ids = df_sorted['simulation_id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=TEST_DATASET_SIZE, random_state=SEED)

    train_df = df_sorted[df_sorted['simulation_id'].isin(train_ids)]
    test_df = df_sorted[df_sorted['simulation_id'].isin(test_ids)]

    train_df.to_csv(training_file_path, index=False)
    print(f"{training_file_path} saved")
    test_df.to_csv(testing_file_path, index=False)
    print(f"{testing_file_path} saved")


def export_dataset(query, conn, file_path):
    df = pd.read_sql_query(query, conn)

    # Sort the DataFrame by 'simulation_id' and then by 'job_start'
    # Need to sort by job start because it is not feasible to learn the behaviour of scheduler
    df_sorted = df.sort_values(by=['simulation_id', 'job_start'])

    # Add an index within each simulation group
    df_sorted['index'] = df_sorted.groupby('simulation_id').cumcount() + 1

    # Group by 'simulation_id' and split
    unique_ids = df_sorted['simulation_id'].unique()

    df = df_sorted[df_sorted['simulation_id'].isin(unique_ids)]

    df.to_csv(file_path, index=False)
    print(f"{file_path} saved")


def export_full_dataset(conn):
    query = "SELECT * FROM simulated_jobs;"
    export_dataset_train_test(query, conn, DEFAULT_TRAINING_FILE_PATH, DEFAULT_TEST_FILE_PATH)


def export_small_dataset(conn):
    # Take only 100 simulations of each size
    query = "SELECT * FROM get_100_simulations_per_group_by_row_number JOIN simulated_jobs USING (simulation_id);"
    export_dataset_train_test(query, conn, DEFAULT_SMALL_TRAINING_FILE_PATH, DEFAULT_SMALL_TEST_FILE_PATH)


def export_extrapolation_dataset(conn):
    query = "SELECT * FROM third_phase_simulated_jobs_extrapolation_x5;"
    export_dataset(query, conn, EXTRAPOLATION_X5_FILE_PATH)

    query = "SELECT * FROM third_phase_simulated_jobs_extrapolation_x10;"
    export_dataset(query, conn, EXTRAPOLATION_X10_FILE_PATH)

    query = "SELECT * FROM third_phase_simulated_jobs_extrapolation_x20;"
    export_dataset(query, conn, EXTRAPOLATION_X20_FILE_PATH)


def main():
    parser = argparse.ArgumentParser(description=f'This tool exports the training dataset and evaluation dataset from database as CSV files.\n'
                                                 f'Training dataset will be saved as [{DEFAULT_TRAINING_FILE_PATH}], test dataset will be saved as [{DEFAULT_TEST_FILE_PATH}].\n'
                                                 f'Proportion: [{1 - TEST_DATASET_SIZE}/{TEST_DATASET_SIZE}]',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Establish the database connection
    conn = establish_database_connection()

    # Export dataset
    export_full_dataset(conn)
    export_small_dataset(conn)
    export_extrapolation_dataset(conn)

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()
