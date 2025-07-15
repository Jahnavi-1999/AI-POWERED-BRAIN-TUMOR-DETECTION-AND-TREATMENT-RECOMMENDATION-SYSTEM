import os
from models.hybrid_vgg16_resnet50 import build_hybrid_model
from models.efficientnet_b2 import build_efficientnet_model
from utils.data_loader import load_dataset
from utils.metrics import plot_training, evaluate_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import json

# --- Configuration ---
DATA_DIR = "data/brain_tumor_classification"
MODEL_TYPE = "hybrid"  # choose: "hybrid" or "efficientnet"
EPOCHS = 10
BATCH_SIZE = 32
SAVE_DIR = "trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load Data ---
(X_train, X_test, y_train, y_test), class_names = load_dataset(DATA_DIR)

# --- Choose Model ---
if MODEL_TYPE == "hybrid":
    model = build_hybrid_model(input_shape=(224, 224, 3), num_classes=len(class_names))
    weights_file = os.path.join(SAVE_DIR, "hybrid_model_weights.h5")
elif MODEL_TYPE == "efficientnet":
    model = build_efficientnet_model(input_shape=(224, 224, 3), num_classes=len(class_names))
    weights_file = os.path.join(SAVE_DIR, "efficientnet_model_weights.h5")
else:
    raise ValueError("Invalid MODEL_TYPE. Use 'hybrid' or 'efficientnet'.")

# --- Compile Model ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
callbacks = [
    ModelCheckpoint(weights_file, save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=3, restore_best_weights=True)
]

# --- Train Model ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# --- Save Full Model ---
model_path = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_model.h5")
save_model(model, model_path)

# --- Save Training History ---
history_path = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_training_history.json")
with open(history_path, 'w') as f:
    json.dump(history.history, f)

# --- Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test Accuracy ({MODEL_TYPE}): {acc*100:.2f}%")

# --- Visualizations ---
plot_training(history)
evaluate_model(model, X_test, y_test, class_names, MODEL_TYPE)
plt.show()