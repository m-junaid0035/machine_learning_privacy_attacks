# attack_mia.py
import os
import numpy as np
from tensorflow import keras

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "output", "models")

# Load the victim model
model_path = os.path.join(MODEL_DIR, "victim_model.h5")
model = keras.models.load_model(model_path)
print("Loaded victim model:", model_path)

# Load MNIST dataset again
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Membership inference attack: threshold-based
def membership_inference(model, x, threshold=0.9):
    preds = model.predict(x, verbose=0)
    confidences = np.max(preds, axis=1)
    members = confidences > threshold
    return members

# Attack evaluation
train_membership = membership_inference(model, x_train)
test_membership = membership_inference(model, x_test)

train_leak_ratio = np.mean(train_membership)
test_leak_ratio = np.mean(test_membership)

print(f"Membership inference attack results:")
print(f"Train data predicted as members: {train_leak_ratio*100:.2f}%")
print(f"Test data predicted as members:  {test_leak_ratio*100:.2f}%")
