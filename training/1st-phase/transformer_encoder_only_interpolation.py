import torch

import apply_model
from transformer_encoder_only_training import TransformerEncoderOnly, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color

purpose="interpolation"

TEST_PATH = '../../dataset-preparation/1st-phase/test_dataset.csv'
input_columns = ['index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']

MODEL_PATH = "generated-models/transformer.pth"
PLOTS_PATH = "plots"

def evaluate(model, model_name, results_directory, batch_size, window_size, window_overlap):
    apply_model.apply_model_to_data(model, model_name, results_directory, TEST_PATH, plot_color, purpose, input_columns, output_columns, batch_size, window_size, window_overlap)


if __name__ == '__main__':
    model: TransformerEncoderOnly = torch.load(MODEL_PATH)
    evaluate(model, model_name, PLOTS_PATH, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE)

