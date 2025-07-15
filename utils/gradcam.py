# utils/gradcam.py

import numpy as np
import cv2
import tensorflow as tf

def generate_gradcam(model, image, layer_name='conv5_block3_out'):
    # Expand dims once, not as a list of two images
    img_input = np.expand_dims(image, axis=0)  # shape: (1, 224, 224, 3)

    # Pass the image as two separate inputs
    inputs = [img_input, img_input]

    # Build submodel for Grad-CAM
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights.numpy())

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Original image for overlay
    img = np.uint8(255 * image)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed
