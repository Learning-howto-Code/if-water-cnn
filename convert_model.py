import tensorflow as tf
import argparse
import os

def convert(model_path, output_path, quantize=False):
    # Detect model type
    if model_path.endswith(".h5"):
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    else:
        # Assume SavedModel directory
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to .tflite")
    parser.add_argument("model_path", help="Path to model (.h5 or SavedModel folder)")
    parser.add_argument("output_path", help="Output .tflite path")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")

    args = parser.parse_args()

    convert(args.model_path, args.output_path, args.quantize)
