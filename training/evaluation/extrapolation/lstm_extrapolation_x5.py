import os

import torch

import apply_model
from lstm_training import BiLSTMModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color, input_columns, output_columns

purpose = "extrapolation"

TEST_FILE_NAME = 'extrapolation_dataset.csv'

MODEL_PATH = "generated-models/lstm.pth"
PLOTS_PATH = "plots"


def evaluate(model, model_name, results_directory, batch_size, window_size, window_overlap, dataset_path):
    apply_model.apply_model_to_data(model, model_name, results_directory, os.path.join(dataset_path, TEST_FILE_NAME), plot_color, purpose, input_columns, output_columns, batch_size, window_size, window_overlap)


if __name__ == '__main__':
    model: BiLSTMModel = torch.load(MODEL_PATH)
    evaluate(model, model_name, PLOTS_PATH, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, TEST_FILE_NAME)
