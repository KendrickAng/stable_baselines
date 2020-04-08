import picamera
import picamera.array

import io
from config import IMAGE_WIDTH, IMAGE_HEIGHT
from PIL import Image
import numpy as np

import logging
from logging import INFO, DEBUG
logging.basicConfig(level=DEBUG, format="%(levelname)s [line %(lineno)d]: %(message)s")
logger = logging.getLogger()
logger.disabled = False

class MyPiCamera:
    """
    This class will be in CarController.py, created at __init__.
    To be used in gym_env.
    """
    def __init__(self):
        logging.info("Creating camera wrapper in gym env")
        self.camera = picamera.PiCamera(resolution=(640, 480), framerate=30)
        # self.stream = io.BytesIO()
        # variables starting with _ are like private vars
        self._input_width = IMAGE_WIDTH
        self._input_height = IMAGE_HEIGHT
        logging.info("input width: {0}, input height: {1}".format(self._input_width, self._input_height))

    def get_image_array(self):
        """
        Captures the current image as seen on camera, converts it to a numpy array and returns it.
        """
        with picamera.array.PiRGBArray(self.camera) as output:
            self.camera.resolution = (640, 480)
            self.camera.capture(output, 'rgb')
            logging.info("Captured image of size {0}x{1}x{2}".format(
                output.array.shape[0], output.array.shape[1], output.array.shape[2]))
            output.truncate(0)
            return output.array
        # self.camera.capture_continuous(self.stream, format='jpeg', use_video_port=True)
        # self.stream.seek(0)
        # image = Image.open(self.stream).convert('RGB').resize((self._input_width, self._input_height), Image.ANTIALIAS)
        # self.stream.seek(0)
        # self.stream.truncate()
        # self.camera.close()

    def quit(self):
        self.camera.close()