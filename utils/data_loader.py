# utils/data_loader.py

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_dataset(data_dir, img_size=224):
    train_dir = os.path.join(data_dir, "Training")
    X, y, class_names = [], [], []
    label_map = {}

    # Get class names and assign numeric labels
    for idx, class_name in enumerate(sorted(os.listdir(train_dir))):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        class_names.append(class_name)
        label_map[class_name] = idx

        # Load all images in the class directory
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # skip non-image files

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image: {img_path}")
                continue

            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(idx)

    X = np.array(X, dtype='float32') / 255.0
    y = to_categorical(y, num_classes=len(class_names))
    
    return train_test_split(X, y, test_size=0.2, random_state=42), class_names
