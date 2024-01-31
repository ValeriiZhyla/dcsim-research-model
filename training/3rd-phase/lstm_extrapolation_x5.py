import torch

import apply_model
from lstm_training import BiLSTMModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color

purpose="extrapolation"

TEST_PATH = '../../dataset_preparation/3rd-phase/extrapolation_dataset_x5.csv'
input_columns = ['index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']

#MODEL_PATH = "../../trained-models/3rd-phase/lstm.pth"
MODEL_PATH = "generated-models/lstm.pth"

model: BiLSTMModel = torch.load(MODEL_PATH)
apply_model.apply_model_to_data(model, model_name, TEST_PATH, plot_color, purpose, input_columns, output_columns, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE)
