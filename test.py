import os
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------
# CONFIG
# -------------------------

MODEL_PATH = "model.tflite"
IMAGE_FOLDER = "production_data"        # folder containing images
IMG_SIZE = (60, 60)            # change to match your model
LABELS = ["water", "no_water"]    # edit if you have more classes

# -------------------------
# LOAD TFLITE MODEL
# -------------------------

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------
# PREDICTION FUNCTION
# -------------------------

def predict_image(img_path):
    # Load image
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    # Convert to array
    arr = np.array(img, dtype=np.float32)

    # Normalize if model expects float input
    if input_details[0]["dtype"] == np.float32:
        arr = arr / 255.0

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Top prediction index
    idx = np.argmax(output)
    confidence = output[idx]

    return idx, confidence

# -------------------------
# LOOP THROUGH ALL IMAGES
# -------------------------

def run_folder(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, filename)
            idx, conf = predict_image(path)

            label = LABELS[idx] if idx < len(LABELS) else f"class_{idx}"
            print(f"{filename}: {label} ({conf:.4f})")

# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":
    run_folder(IMAGE_FOLDER)
