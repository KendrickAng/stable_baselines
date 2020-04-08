# import numpy as np
# import io
# import picamera
# import time

# from PIL import Image
#from tflite_runtime.interpreter import Interpreter

from stable_baselines.sac.policies import CnnPolicy
from stable_baselines import SAC
from stable_baselines.common.callbacks import CheckpointCallback
from gym_env import AutoDriftEnv
from sac_model import SacModel

# import socket
# import pickle

import os
import logging
from logging import INFO

logging.basicConfig(level=INFO, format="%(levelname)s [line %(lineno)d]: %(message)s")
logger = logging.getLogger()
logger.disabled = False

class DriveAgent:
    """
    Python 3. The rest of the files are in Python 2.
    """
    def __init__(self):
        logger.info(os.getcwd())
        #self._interpreter = Interpreter("./converted_model.tflite")
        #self._interpreter.allocate_tensors()
        #print(self._interpreter.get_input_details())
        #print(self._interpreter.get_output_details())
        #_, self._input_height, self._input_width, _ = self._interpreter.get_input_details()[0]['shape']

        self.env = AutoDriftEnv(const_throttle=0.5)
        # self.model = SacModel(policy=CnnPolicy, env=self.env)
        self.model = SAC(policy=CnnPolicy, env=self.env)

        # self._input_height = IMAGE_HEIGHT
        # self._input_width = IMAGE_WIDTH
        # print(self._input_height)
        # print(self._input_width)

        # self._socket = socket.socket()
        # socket_addr = ('127.0.0.1', 8888)
        # UNCOMMENT THIS
        #self._socket.connect(socket_addr)

        self.main()

    def main(self):
        try:
            # Save a checkpoint every 1000 steps
            # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/callbacks.py
            checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs', name_prefix='rl_model', verbose=2)
            self.model.learn(total_timesteps=5000, log_interval=4, callback=checkpoint_callback)

        # input_details = self._interpreter.get_input_details()
        # output_details = self._interpreter.get_output_details()
        # with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
        #     # camera.vflip = True
        #     # camera.start_preview()
        #     try:
        #         stream = io.BytesIO()
        #         for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
        #             stream.seek(0)
        #             image = Image.open(stream).convert('RGB').resize((self._input_width, self._input_height), Image.ANTIALIAS)
        #             start_time = time.time()
        #
        #             img = np.asarray(image)
        #             img = img[np.newaxis, ...] # what's this for?
        #             input_data = np.array(img, dtype=np.float32)
        #             print(input_data.shape)
        #             print(input_data)
        #             #self._interpreter.set_tensor(input_details[0]['index'], input_data)
        #
        #             #self._interpreter.invoke()
        #             #output_data = self._interpreter.get_tensor(output_details[0]['index'])[0]
        #             # TEMP FIX
        #             output_data = None
        #             # time_taken_ms = (time.time() - start_time) * 1000
        #             # print(f'output_data:{output_data}, time_taken:{time_taken_ms}ms')
        #             # camera.annotate_text = str(output_data) + ", " + str(time_taken_ms)
        #             stream.seek(0)
        #             stream.truncate()
        #
        #             data = []
        #             data.append(output_data)
        #             data_string = pickle.dumps(data, protocol=1)
        #             self._socket.send(data_string)
        #
        except KeyboardInterrupt:
            print("DriveAgent: Ctrl-C")
        finally:
            # camera.stop_preview()
            self.env.close()
            print("DriveAgent: environment closed, done")

if __name__ == "__main__":
    DriveAgent()

