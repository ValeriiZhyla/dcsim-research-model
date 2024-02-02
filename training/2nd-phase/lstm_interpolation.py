import torch

import apply_model
from lstm_training import BiLSTMModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color, input_columns, output_columns

purpose = "interpolation"

TEST_PATH = '../../dataset_preparation/2nd-phase/test_dataset.csv'

MODEL_PATH = "generated-models/lstm.pth"
PLOTS_PATH = "plots"


def evaluate(model, model_name, results_directory, batch_size, window_size, window_overlap):
    apply_model.apply_model_to_data(model, model_name, results_directory, TEST_PATH, plot_color, purpose, input_columns, output_columns, batch_size, window_size, window_overlap)


if __name__ == '__main__':
    model: BiLSTMModel = torch.load(MODEL_PATH)
    evaluate(model, model_name, PLOTS_PATH, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE)
