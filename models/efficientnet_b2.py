# models/efficientnet_b2.py

from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

def build_efficientnet_model(input_shape=(224, 224, 3), num_classes=3):
    base_model = EfficientNetB2(include_top=False, weights='imagenet', input_shape=input_shape)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
