import tensorflow as tf
import argparse
import os

def convert(model_path, output_path, quantize=False):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert model
    tflite_model = converter.convert()

    # Write output .tflite file
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .keras model to .tflite")
    parser.add_argument("model_path", help="Path to input .keras model")
    parser.add_argument("output_path", help="Path to output .tflite file")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable dynamic range quantization")

    args = parser.parse_args()

    # Run conversion
    convert(args.model_path, args.output_path, args.quantize)
