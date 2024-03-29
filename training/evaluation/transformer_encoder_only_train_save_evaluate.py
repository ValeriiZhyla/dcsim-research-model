import transformer_encoder_only_training
from training import commons
import transformer_encoder_only_extrapolation_x5


class Hyperparameters:
    epochs = 0
    window_size = 0
    window_overlap = 0
    batch_size = 0
    hidden_size = 0
    heads = 0
    encoders = 0
    decoders = 0 # No decoders

    def __init__(self, epochs, window_size, window_overlap, batch_size, hidden_size, heads, encoders):
        self.epochs = epochs
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.encoders = encoders
        self.decoders = 0  # No decoders



hyperparameter_combinations = [
    Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),
]


for hp in hyperparameter_combinations:
    if commons.directory_name_with_hyperparameters_already_exists(transformer_encoder_only_training.model_name, hp.epochs, hp.window_size, hp.window_overlap, hp.batch_size, hp.hidden_size, heads=hp.heads, encoders=hp.encoders):
        continue
    else:
        # Train
        model: transformer_encoder_only_training.TransformerEncoderOnly = transformer_encoder_only_training.train_and_evaluate_model(num_epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                                                  batch_size=hp.batch_size, hidden_size=hp.hidden_size, nheads=hp.heads, encoder_layers=hp.encoders)
        # Save
        directory: str = commons.save_model_and_get_directory(model, transformer_encoder_only_training.model_name, epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                              batch_size=hp.batch_size, hidden_size=hp.hidden_size, heads=hp.heads, encoders=hp.encoders)
        print(f"Model trained and saved in {directory}")

        # Evaluate: Extrapolation
        transformer_encoder_only_extrapolation_x5.evaluate(model, transformer_encoder_only_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap)
        print(f"Model evaluated: extrapolation")
        print(f"Model {directory} processed")
        print("===============================")
