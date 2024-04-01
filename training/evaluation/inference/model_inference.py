import json
import time

import torch

import commons
from evaluation.gru_training import BiGRUModel
from evaluation.lstm_training import BiLSTMModel
from evaluation.transformer_encoder_only_training import TransformerEncoderOnly

models = {"gru": ['../generated-models/GRU_500_300ep_150wd_0wo_64bat_128hs_1la/GRU.pth', '../generated-models/GRU_500_300ep_150wd_0wo_64bat_128hs_1la/hyperparameters.json'],
          "lstm": ['../generated-models/LSTM_128-500_300ep_100wd_0wo_128bat_128hs_1la/LSTM.pth', '../generated-models/LSTM_128-500_300ep_100wd_0wo_128bat_128hs_1la/hyperparameters.json'],
          "transformer": ['../generated-models/Transformer_500_500ep_100wd_0wo_256bat_16hs_4hd_1enc_0dec/Transformer.pth', '../generated-models/Transformer_500_500ep_100wd_0wo_256bat_16hs_4hd_1enc_0dec//hyperparameters.json'],
          }
datasets = {"5": "../../../dataset_preparation/evaluation/inference/inference_5",
            "10": "../../../dataset_preparation/evaluation/inference/inference_10",
            "20": "../../../dataset_preparation/evaluation/inference/inference_20",
            "50": "../../../dataset_preparation/evaluation/inference/inference_50",
            "100": "../../../dataset_preparation/evaluation/inference/inference_100",
            "250": "../../../dataset_preparation/evaluation/inference/inference_250",
            "500": "../../../dataset_preparation/evaluation/inference/inference_500",
            "1000": "../../../dataset_preparation/evaluation/inference/inference_1000",
            "1500": "../../../dataset_preparation/evaluation/inference/inference_1500",
            "2000": "../../../dataset_preparation/evaluation/inference/inference_2000",
            "10000": "../../../dataset_preparation/evaluation/inference/inference_10000",
}

input_columns = ['simulation_length', 'index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']


for model_name in models:
    model_path = models[model_name][0]
    model_hyperparameters_path = models[model_name][1]
    with open(model_hyperparameters_path, 'r') as file:
        model_hyperparameters = json.load(file)
    model = torch.load(model_path)
    for dataset_name in datasets:
        dataset_path = datasets[dataset_name]
        start_preparation = time.time()
        test_loader, test_scalers = commons.load_test_data(dataset_path, input_columns, output_columns, model_hyperparameters["batch_size"], model_hyperparameters["window_size"], model_hyperparameters["window_overlap"])
        end_preparation = time.time()

        # Define the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        model.to(device)

        # Evaluate the model with test data
        start_inference = time.time()
        predictions_array, actual_values_array, actual_inputs_array = commons.evaluate_model_get_predictions_and_actual_values(model, test_loader, device)
        end_inference = time.time()

        print(f"Model {model_name} processed sequence with {dataset_name} items. Preparation: {end_preparation - start_preparation}, Inference: {end_inference - start_inference}")




