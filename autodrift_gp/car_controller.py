import cv2
import numpy as np
import webcolors
import time
from config import INPUT_DIM, MIN_THROTTLE, MAX_THROTTLE, CRASH_SPEED_WEIGHT, REWARD_CRASH, THROTTLE_REWARD_WEIGHT
from camera import MyPiCamera
from car_server import NavInterface

import logging
from logging import INFO, DEBUG
logging.basicConfig(level=INFO, format="%(levelname)s [%(filename)s line %(lineno)d]: %(message)s")
logger = logging.getLogger()
logger.disabled = False

class CarController:
    """
    Interface for communicating and sending actions to the ROS car. Also calculating rewards, checking if game over
    (car exceeded track bounds). Its a mess.
    """
    def __init__(self):
        # sensors
        self.camera = MyPiCamera()
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.last_throttle = 0.0

        # interface to NavController
        self.nav_interface = NavInterface()

        self.steering_angle = 0.0
        self.current_step = 0
        self.speed = 0
        self.steering = None

    """
    INTERFACE WITH GYM_ENV
    """
    # Tell the car to move with steer and throttle.
    def take_action(self, action):
        """
        :param action: ([float]) [Steering, throttle]
        """
        # update steering angle
        self.steering = action[0]
        self.last_throttle = action[1]
        self.current_step += 1

        # TODO: Send controls to the car
        self.nav_interface.send_data(action)

    # Get observation of the environment.
    def observe(self):
        """
        :return: (observation, reward, done, info)
        """
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        # update model's image with image being updated in real-time
        self.image_array = self.camera.get_image_array()

        done = self.is_game_over()

        return self.image_array, self.calc_reward(done), done, {}

    def calc_reward(self, done):
        """
        Compute reward:
        - +1 life bonus for each step + throttle bonus
        - -10 crash penalty - penalty for large throttle during a crash

        :param done: (bool)
        :return: (float)
        """
        if done:
            # penalize the agent for getting off the road fast
            norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle
        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        return 1 + throttle_reward

    # reset all states (numpy arrays etc) to new
    def reset(self):
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.last_throttle = 0.0

        self.steering_angle = 0.0
        self.current_step = 0
        self.speed = 0
        self.steering = None

        # TODO: Send the car an action of (0 throttle, 0 steering)
        self.nav_interface.send_data(np.array([0.0, 0.0]))

    def quit(self):
        self.nav_interface.quit()
        self.camera.quit()

    """
    CLASS-SPECIFIC FUNCTIONS (no interaction with other classes)
    """
    def is_game_over(self):
        """
        Game over = cross track boundary = when the dominant color (Red or Yellow or Blue) crosses a defined threshold.
        Use k means clustering to find the dominant color in the image.
        https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
        """
        pixels = np.float32(self.image_array.reshape(-1, 3))
        # cluster image pixels into 3 (color) clusters, 0 to 2
        n_colors = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

        # (ndarray) counts = the number of times each unique item appears in ar.
        _, counts = np.unique(labels,return_counts=True)
        # tuple of (r,g,b) for dominant color in image
        dominant_rgb = palette[np.argmax(counts)]
        actual_color_name , closest_colour_name = get_colour_name(dominant_rgb)

        # black, silver, gray, white, maroon, red, purple, fuchsia, green, lime, olive, yellow, navy, blue, teal, aqua
        # out of track when camera detects too much of one color
        if actual_color_name:
            a = actual_color_name
            logger.info("{0} (actual) detected on camera".format(a))
            return a=='red' or a=='yellow' or a=='blue'
        else:
            c = closest_colour_name
            logger.info("{0} (closest) detected on camera".format(c))
            return c=='red' or c=='yellow' or c=='blue'


# https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
def closest_colour(requested_colour):
    """
    Calculates euclidean distance to map a (r,g,b) color to a color name as given in CSS2 specifications.
    https://www.w3.org/TR/css-color-3/ - see list of basic colors
    :param requested_colour:
    :return:
    """
    min_colours = {}
    # print(webcolors.CSS2_HEX_TO_NAMES)
    for key, name in webcolors.CSS2_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name