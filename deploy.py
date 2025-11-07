from tensorflow import keras
from picamera2 import Picamera2
from time import sleep, strftime, time
import os
import cv2
# loads model
model = keras.models.load_model("model.keras")
# starts camera as a video so it can go faster
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (224, 224)}, buffer_count=4)
picam2.configure(config)
picam2.start()



def test_pic():
    frame = picam2.capture_array()
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    print(prediction)
    return prediction

while True:
    sleep(0.2)
    test_pic()
#steps

#take an image
#pass that image into the model
#delete the image