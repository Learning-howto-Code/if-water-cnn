from picamera2 import Picamera2
from time import sleep
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import sys
from pi5neo import Pi5Neo


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

#sets up led ring
SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)

# Fill the strip with white (R,G,B = 255,255,255)
neo.fill_strip(255, 255, 255)
neo.update_strip()  # commit/send to LEDs


def test_pic():
    frame = picam2.capture_array()

    # Convert BGR→RGB, then resize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))

    # Normalize
    img = img.astype("float32") / 255.0

    # Add batch dimension → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])
    print("closer to 0 = no water, closer to 1 = water")
    if prediction < 0.5:
        print(" No Water Detected")
    else:
        print("Water Detected")
    print(prediction)
    
    return prediction

x = 0
while x < 5:
    sleep(0.2)
    test_pic()
    x += 1
neo.fill_strip(0, 0, 0)
neo.update_strip()  # commit/send to LEDs