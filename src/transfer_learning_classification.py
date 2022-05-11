from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import applications


def create_model(weights="imagenet", n_classes=131):
    """
    Create a model and add a custom output layer. Also add the weights to the model if required.
    """
    base_model = applications.ResNet50(weights=weights, include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(512, activation="relu")(x)
    # and a fully connected output/classification layer
    predictions = Dense(n_classes, activation="softmax")(x)
    # create the full network so we can train on it
    return Model(inputs=base_model.input, outputs=predictions)
