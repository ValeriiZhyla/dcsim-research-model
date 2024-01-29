import torch

import apply_model
from transformer_training_encoder_only import TransformerModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color

purpose="interpolation"

TEST_PATH = '../../dataset-preparation/2nd-phase/test_dataset.csv'
input_columns = ['index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']

#MODEL_PATH = "../../trained-models/2nd-phase/transformer.pth"
MODEL_PATH = "generated-models/transformer.pth"

model: TransformerModel = torch.load(MODEL_PATH)
apply_model.apply_model_to_data(model, model_name, TEST_PATH, plot_color, purpose, input_columns, output_columns, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE)
