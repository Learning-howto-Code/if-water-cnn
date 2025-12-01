import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix

MODEL_PATH = "model.tflite"
IMAGE_FOLDER = "images"

IMG_SIZE = (224, 224)
LABELS = ["clean", "dirty"]     # adjust as needed
TRUE_CLASS = 0                 # all images belong to "clean" class

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)
    if input_details[0]["dtype"] == np.float32:
        arr = arr / 255.0

    arr = np.expand_dims(arr, 0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    return np.argmax(output)

def run_folder(folder):
    y_true = []
    y_pred = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, filename)

            pred = predict_image(path)

            y_true.append(TRUE_CLASS)
            y_pred.append(pred)

            print(f"{filename}: predicted {pred}")

    return y_true, y_pred

if __name__ == "__main__":
    y_true, y_pred = run_folder(IMAGE_FOLDER)

    cm = confusion_matrix(y_true, y_pred)
    print("\nCONFUSION MATRIX")
    print(cm)
