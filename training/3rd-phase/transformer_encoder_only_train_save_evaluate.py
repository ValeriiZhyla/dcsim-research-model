import transformer_encoder_only_training
from training import commons
import transformer_encoder_only_interpolation
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
    # Hyperparameters(epochs=50, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=1),  # test
    #
    # # Check different hidden sizes
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=4, heads=1, encoders=1),
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=1),
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1), # 16 hs is better
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=32, heads=1, encoders=1),
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=64, heads=1, encoders=1),
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=128, heads=1, encoders=1),
    #
    # # Check different window sizes
    # Hyperparameters(epochs=120, window_size=25, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=50, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1), # 100 wd is better
    # Hyperparameters(epochs=120, window_size=125, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=150, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=175, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=200, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    #
    # # Check different window overlaps
    # Hyperparameters(epochs=120, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1), # no overlap is better
    # Hyperparameters(epochs=120, window_size=100, window_overlap=10, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=100, window_overlap=20, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=100, window_overlap=30, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=100, window_overlap=40, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=120, window_size=100, window_overlap=50, batch_size=128, hidden_size=16, heads=1, encoders=1),
    #
    # # Check different encoders
    # Hyperparameters(epochs=150, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1), # 1 encoder is better
    # Hyperparameters(epochs=150, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=2),
    # Hyperparameters(epochs=150, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=4),
    # Hyperparameters(epochs=150, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=8),
    # Hyperparameters(epochs=150, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=16),
    #
    # # Check different nheads
    # Hyperparameters(epochs=170, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=170, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=2, encoders=1),
    # Hyperparameters(epochs=170, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=4, encoders=1), # 4 heads best result
    # Hyperparameters(epochs=170, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=8, encoders=1),
    #
    # # Check different batch sizes
    # Hyperparameters(epochs=200, window_size=100, window_overlap=0, batch_size=64, hidden_size=16, heads=4, encoders=1),
    # Hyperparameters(epochs=200, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=4, encoders=1),
    # Hyperparameters(epochs=200, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1), # 256 is the best

    # Train longer
    Hyperparameters(epochs=101, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, heads=4, encoders=1),  # 256 is the best
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

        # Evaluate: Interpolation
        transformer_encoder_only_interpolation.evaluate(model, transformer_encoder_only_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap)
        print(f"Model evaluated: interpolation")

        # Evaluate: Extrapolation
        transformer_encoder_only_extrapolation_x5.evaluate(model, transformer_encoder_only_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap)
        print(f"Model evaluated: extrapolation")
        print(f"Model {directory} processed")
        print("===============================")
