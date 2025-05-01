import tensorflow as tf
from tensorflow import keras


# Path to your existing Keras model
keras_model_path = "C:/Users/Sushil Kumar Singh/Desktop/Model/keras_model.h5"
tflite_model_path = "C:/Users/Sushil Kumar Singh/Desktop/Model/converted_model.tflite"

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert the Keras model to a SavedModel
saved_model_dir = "/Users/Sushil Kumar Singh/Desktop/Model/saved_model"
model.save(saved_model_dir)

# Reload the model as a TensorFlow SavedModel
loaded_model = tf.saved_model.load(saved_model_dir)

# Select the default signature function
inference_func = loaded_model.signatures["serving_default"]

# Convert to TensorFlow Lite with a single signature
converter = tf.lite.TFLiteConverter.from_concrete_functions([inference_func])
tflite_model = converter.convert()

# Save the converted TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model successfully converted and saved to {tflite_model_path}")
