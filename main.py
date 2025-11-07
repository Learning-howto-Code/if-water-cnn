from picamera2 import Picamera2
from time import sleep, strftime, time
from pi5neo import Pi5Neo
import os
import cv2

# Image folder
img_folder = "/home/jake/Downloads/data_collection/water_start_end/toilet"

# SPI setup for NeoPixel
SPI_DEVICE = '/dev/spidev0.0'
SPI_SPEED_KHZ = 800
pin = 30
neo = Pi5Neo(SPI_DEVICE, pin, SPI_SPEED_KHZ)
 
def take_pic():
    neo.fill_strip(10, 10, 10) #sets LED's to white and a little dimmer
    neo.update_strip() #sets color
    print("light on")
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    picam2.configure(config) #sets configuration
    picam2.start()
    print("activated cam")
    sleep(2)    #watis for cam to start
    print("waited 2 sec") #cam is now ready
    frame_count=0
    start_time = time() 
    while time() - start_time < 600:  #takes images for 20 seconds
        frame = picam2.capture_array()  #uses capture array funtion instead
        frame_count += 1
#
        
        filename = f"{img_folder}/img_{strftime('%Y%m%d_%H%M%S')}_{frame_count}.jpg"
        cv2.imwrite(filename, frame)

    print("Captured", frame_count, "frames in", round(time()-start_time, 2), "seconds")


   

take_pic()
