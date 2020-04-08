import numpy as np
import io
import picamera
import time

from PIL import Image
#from tflite_runtime.interpreter import Interpreter

import socket
import pickle

import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class DriveAgent:
    def __init__(self):
        logger.info(os.getcwd())
        #self._interpreter = Interpreter("./converted_model.tflite")
        #self._interpreter.allocate_tensors()

        #print(self._interpreter.get_input_details())
        #print(self._interpreter.get_output_details())
        #_, self._input_height, self._input_width, _ = self._interpreter.get_input_details()[0]['shape']
        self._input_height = 160
        self._input_width = 80
        print(self._input_height)
        print(self._input_width)

        self._socket = socket.socket()
        socket_addr = ('127.0.0.1', 8888)
        # UNCOMMENT THIS
        #self._socket.connect(socket_addr)

        self.main()

    def main(self):
        #input_details = self._interpreter.get_input_details()
        #output_details = self._interpreter.get_output_details()
        with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
            # camera.vflip = True
            # camera.start_preview()
            try:
                stream = io.BytesIO()
                for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    stream.seek(0)
                    image = Image.open(stream).convert('RGB').resize((self._input_width, self._input_height), Image.ANTIALIAS)
                    start_time = time.time()

                    img = np.asarray(image)
                    img = img[np.newaxis, ...] # what's this for?
                    input_data = np.array(img, dtype=np.float32)
                    print(input_data.shape)
                    print(input_data)
                    #self._interpreter.set_tensor(input_details[0]['index'], input_data)

                    #self._interpreter.invoke()
                    #output_data = self._interpreter.get_tensor(output_details[0]['index'])[0]
                    # TEMP FIX
                    output_data = None
                    # time_taken_ms = (time.time() - start_time) * 1000
                    # print(f'output_data:{output_data}, time_taken:{time_taken_ms}ms')
                    # camera.annotate_text = str(output_data) + ", " + str(time_taken_ms)
                    stream.seek(0)
                    stream.truncate()

                    data = []
                    data.append(output_data)
                    data_string = pickle.dumps(data, protocol=1)
                    self._socket.send(data_string)

            except KeyboardInterrupt:
                print("DriveAgent: Ctrl-C")
            finally:
                # camera.stop_preview()
                print("DriveAgent: done")

if __name__ == "__main__":
    DriveAgent()

