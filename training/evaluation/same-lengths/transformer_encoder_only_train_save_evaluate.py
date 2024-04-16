import transformer_encoder_only_training
from evaluation import transformer_encoder_only_interpolation
from training import commons

class Hyperparameters:
    prefix = ""
    dataset_directory = ""
    epochs = 0
    window_size = 0
    window_overlap = 0
    batch_size = 0
    hidden_size = 0
    heads = 0
    encoders = 0
    decoders = 0 # No decoders

    def __init__(self, prefix, dataset_directory, epochs, window_size, window_overlap, batch_size, hidden_size, heads, encoders):
        self.prefix = prefix
        self.dataset_directory = dataset_directory
        self.epochs = epochs
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.encoders = encoders
        self.decoders = 0  # No decoders



hyperparameter_combinations = [
    Hyperparameters("1", "../../../dataset_preparation/evaluation/same_length/1", epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),
    Hyperparameters("10", "../../../dataset_preparation/evaluation/same_length/10", epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),
    Hyperparameters("20", "../../../dataset_preparation/evaluation/same_length/20", epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),
    Hyperparameters("30", "../../../dataset_preparation/evaluation/same_length/30", epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),
    Hyperparameters("40", "../../../dataset_preparation/evaluation/same_length/40", epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),
    Hyperparameters("50", "../../../dataset_preparation/evaluation/same_length/50", epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),
]


for hp in hyperparameter_combinations:
    if commons.directory_name_with_hyperparameters_already_exists(transformer_encoder_only_training.model_name, hp.epochs, hp.window_size, hp.window_overlap, hp.batch_size, hp.hidden_size, heads=hp.heads, encoders=hp.encoders, prefix=hp.prefix):
        continue
    else:
        # Train
        model: transformer_encoder_only_training.TransformerEncoderOnly = transformer_encoder_only_training.train_and_evaluate_model(num_epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                                                  batch_size=hp.batch_size, hidden_size=hp.hidden_size, nheads=hp.heads, encoder_layers=hp.encoders, dataset_directory=hp.dataset_directory)
        # Save
        directory: str = commons.save_model_and_get_directory(model, transformer_encoder_only_training.model_name, epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                              batch_size=hp.batch_size, hidden_size=hp.hidden_size, heads=hp.heads, encoders=hp.encoders, prefix=hp.prefix)
        print(f"Model trained and saved in {directory}")
        # Evaluate: Interpolation
        transformer_encoder_only_interpolation.evaluate(model, transformer_encoder_only_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap, hp.dataset_directory)
        print(f"Model evaluated: interpolation")


        # Evaluate: Extrapolation
        # transformer_encoder_only_extrapolation_x5.evaluate(model, transformer_encoder_only_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap, hp.dataset_directory)
        # print(f"Model evaluated: extrapolation")
        print(f"Model {directory} processed")
        print("===============================")
