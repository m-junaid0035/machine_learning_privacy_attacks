# defense_dp.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_privacy as tfp

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Build CNN model (same as before)
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    return model

model = build_model()

# Differential Privacy Optimizer
optimizer = tfp.dp_optimizer_keras.DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=128,
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train DP model
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=6,
    batch_size=128,
    verbose=2
)

# Evaluate
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTrain accuracy (DP model): {train_acc:.4f}")
print(f"Test accuracy (DP model):  {test_acc:.4f}")

# Save DP model
model_path = os.path.join(MODEL_DIR, "dp_model.h5")
model.save(model_path)
print("Saved DP model to:", model_path)

# Plot training graphs
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("DP Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("DP Accuracy")
plt.legend()

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "dp_training_history.png")
plt.savefig(plot_path)
print("Saved DP training plot to:", plot_path)
