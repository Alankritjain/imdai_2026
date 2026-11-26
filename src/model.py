"""Define the compact milestone CNN for continuous steering regression."""

from config import IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE

ARCHITECTURE = """Input: (80, 160, 3), RGB pixels in [0, 1]
Conv2D: 16 filters, 5 x 5, stride 2, ReLU
Conv2D: 32 filters, 3 x 3, stride 2, ReLU
Conv2D: 48 filters, 3 x 3, stride 2, ReLU
GlobalAveragePooling2D
Dense: 32, ReLU
Dropout: 0.2
Dense: 1, linear (continuous steering value)"""


def build_model(learning_rate=LEARNING_RATE):
    """Build and compile the untrained CNN; TensorFlow is imported only here."""
    try:
        from tensorflow import keras
    except ImportError as error:
        raise ImportError(
            "TensorFlow is optional. In a Python 3.11 environment, run "
            "python -m pip install -r requirements-model.txt to use the model."
        ) from error
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        # Strides reduce spatial size while convolution learns visual features.
        layers.Conv2D(16, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, strides=2, activation="relu"),
        layers.Conv2D(48, 3, strides=2, activation="relu"),
        # Pooling keeps parameter count lower than flattening all feature pixels.
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),  # Regularization during training only.
        layers.Dense(1, activation="linear"),  # Regression, not classification.
    ], name="lightweight_steering_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def main():
    try:
        model = build_model()
    except (ImportError, OSError) as error:
        raise SystemExit(str(error)) from error
    model.summary()
    print("Fresh architecture inspection only: these parameters are untrained. "
          "Use src.predict to load the saved trained checkpoint.")


if __name__ == "__main__":
    main()
