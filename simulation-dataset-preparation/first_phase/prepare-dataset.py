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
DEFAULT_TRAINING_FILE_PATH = "../first_phase/train_dataset.csv"
DEFAULT_TEST_FILE_PATH = "../first_phase/test_dataset.csv"
DEFAULT_SMALL_TRAINING_FILE_PATH = "../first_phase/train_dataset_small.csv"
DEFAULT_SMALL_TEST_FILE_PATH = "../first_phase/test_dataset_small.csv"

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


def export_dataset(query, conn, training_file_path, testing_file_path):
    df = pd.read_sql_query(query, conn)

    # Sort the DataFrame by 'simulation_id' and then by 'job_start'
    # Need to sort by job start because it is not feasible to learn the behaviour of scheduler
    df_sorted = df.sort_values(by=['simulation_id', 'job_start'])

    # Group by 'simulation_id' and split
    unique_ids = df_sorted['simulation_id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=TEST_DATASET_SIZE, random_state=SEED)

    train_df = df_sorted[df_sorted['simulation_id'].isin(train_ids)]
    test_df = df_sorted[df_sorted['simulation_id'].isin(test_ids)]

    train_df.to_csv(training_file_path, index=False)
    test_df.to_csv(testing_file_path, index=False)


def export_full_dataset(conn):
    query = "SELECT * FROM simulated_jobs;"
    export_dataset(query, conn, DEFAULT_TRAINING_FILE_PATH, DEFAULT_TEST_FILE_PATH)

def export_small_dataset(conn):
    # Take only 100 simulations of each size
    query = "SELECT * FROM get_100_simulations_per_group_by_row_number JOIN simulated_jobs USING (simulation_id);"
    export_dataset(query, conn, DEFAULT_SMALL_TRAINING_FILE_PATH, DEFAULT_SMALL_TEST_FILE_PATH)

def main():
    parser = argparse.ArgumentParser(description=f'This tool exports the training dataset and evaluation dataset from database as CSV files.\n'
                                                 f'Training dataset will be saved as [{DEFAULT_TRAINING_FILE_PATH}], test dataset will be saved as [{DEFAULT_TEST_FILE_PATH}].\n'
                                                 f'Proportion: [{1-TEST_DATASET_SIZE}/{TEST_DATASET_SIZE}]',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Establish the database connection
    conn = establish_database_connection()

    # Export dataset
    export_full_dataset(conn)
    export_small_dataset(conn)

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()
