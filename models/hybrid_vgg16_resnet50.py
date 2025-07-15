# models/hybrid_vgg16_resnet50.py

from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, concatenate
from tensorflow.keras.models import Model

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=4):
    # Separate inputs for both VGG16 and ResNet50
    vgg_input = Input(shape=input_shape)
    resnet_input = Input(shape=input_shape)

    # Load pre-trained models
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=vgg_input)
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=resnet_input)

    # Freeze convolutional layers
    for layer in vgg.layers:
        layer.trainable = False
    for layer in resnet.layers:
        layer.trainable = False

    # Feature maps
    vgg_out = GlobalAveragePooling2D()(vgg.output)
    resnet_out = GlobalAveragePooling2D()(resnet.output)

    # Merge features
    merged = concatenate([vgg_out, resnet_out])
    dense1 = Dense(128, activation='relu')(merged)
    output = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs=[vgg_input, resnet_input], outputs=output)
    return model
