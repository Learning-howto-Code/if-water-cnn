from picamera2 import Picamera2
from time import sleep
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite


interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


h = input_details[0]["shape"][1]
w = input_details[0]["shape"][2]

#sets up camera 
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (w, h)}, buffer_count=4)
picam2.configure(config)
picam2.start()


def test_pic():
    frame = picam2.capture_array()

    # Resize + normalize
    img = cv2.resize(frame, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Send input to TFLite model
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    # Read output
    prediction = interpreter.get_tensor(output_details[0]["index"])
    print(prediction)
    return prediction
x = 0
while x < 5:
    sleep(0.2)
    test_pic()
    x += 1