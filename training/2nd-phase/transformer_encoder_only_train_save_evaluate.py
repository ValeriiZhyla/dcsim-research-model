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
    # Hyperparameters(epochs=10, window_size=200, window_overlap=100, batch_size=128, hidden_size=8, heads=1, encoders=1),  # test
    # Hyperparameters(epochs=100, window_size=200, window_overlap=100, batch_size=128, hidden_size=8, heads=2, encoders=2),  # less layers
    # Hyperparameters(epochs=100, window_size=200, window_overlap=100, batch_size=128, hidden_size=50, heads=2, encoders=2),  # medium layers
    # Hyperparameters(epochs=100, window_size=200, window_overlap=100, batch_size=128, hidden_size=100, heads=2, encoders=2),  # more layers
    # Hyperparameters(epochs=100, window_size=100, window_overlap=50, batch_size=128, hidden_size=100, heads=2, encoders=2),  # more layers, smaller window
    # Hyperparameters(epochs=100, window_size=50, window_overlap=40, batch_size=128, hidden_size=100, heads=2, encoders=2),  # more layers, even smaller window

    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=2, encoders=2), # no overlap
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=2, encoders=1),  # no overlap, one encoder
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=2, encoders=4),  # no overlap, 4 encoders

    # # Check different hidden sizes
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=4, heads=1, encoders=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=1), # <== Best hidden size is 8
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, heads=1, encoders=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=32, heads=1, encoders=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=64, heads=1, encoders=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=128, heads=1, encoders=1),
    #
    # # Check different batch sizes
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=32, hidden_size=8, heads=1, encoders=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=64, hidden_size=8, heads=1, encoders=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=1),  # <== Best batch size is 128
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=8, heads=1, encoders=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=512, hidden_size=8, heads=1, encoders=1),
    #
    # # Check different encoders
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=1), # <== Best encoders is 1
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=2),
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=4),
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=8),
    #
    # # Check different heads
    # Hyperparameters(epochs=550, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=1, encoders=1),
    # Hyperparameters(epochs=550, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=2, encoders=1),
    # Hyperparameters(epochs=550, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=4, encoders=1), # <== Best heads is 4
    # Hyperparameters(epochs=550, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=8, encoders=1),
    #
    # # Check different window sizes
    # Hyperparameters(epochs=550, window_size=50, window_overlap=0, batch_size=128, hidden_size=8, heads=4, encoders=1),
    # Hyperparameters(epochs=550, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=4, encoders=1),  # <== Best window size is 100
    # Hyperparameters(epochs=550, window_size=150, window_overlap=0, batch_size=128, hidden_size=8, heads=4, encoders=1),
    #
    # # Check different window overlaps
    # Hyperparameters(epochs=250, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, heads=4, encoders=1),
    # Hyperparameters(epochs=250, window_size=100, window_overlap=10, batch_size=128, hidden_size=8, heads=4, encoders=1),
    # Hyperparameters(epochs=250, window_size=100, window_overlap=20, batch_size=128, hidden_size=8, heads=4, encoders=1),
    # Hyperparameters(epochs=250, window_size=100, window_overlap=30, batch_size=128, hidden_size=8, heads=4, encoders=1), # <== Best window overlap is 30 so far
    # Hyperparameters(epochs=250, window_size=100, window_overlap=40, batch_size=128, hidden_size=8, heads=4, encoders=1),
    # Hyperparameters(epochs=250, window_size=100, window_overlap=50, batch_size=128, hidden_size=8, heads=4, encoders=1),
    # Hyperparameters(epochs=250, window_size=100, window_overlap=60, batch_size=128, hidden_size=8, heads=4, encoders=1),

    # Best model:     Hyperparameters(epochs=250, window_size=100, window_overlap=30, batch_size=128, hidden_size=8, heads=4, encoders=1), but no huge difference compared to 0 overlap, 1.5% of R2
    Hyperparameters(epochs=101, window_size=100, window_overlap=30, batch_size=128, hidden_size=8, heads=4, encoders=1)
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
