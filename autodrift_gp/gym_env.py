import gym
import time
import numpy as np
from gym import spaces
from car_controller import CarController

from config import INPUT_DIM, MAX_STEERING, MAX_THROTTLE, MIN_THROTTLE

import logging
from logging import INFO, DEBUG
logging.basicConfig(level=DEBUG, format="%(levelname)s [line %(lineno)d]: %(message)s")
logger = logging.getLogger()
logger.disabled = False

class AutoDriftEnv(gym.Env):
    """
    Custom gym environment for the Autodrift Grand Prix.
    For now, assume that camera image input is (80, 160, 3) (height, width, depth).
    Command history is not supported yet, and with it jerk penalty is unsupported.
    Frame skip is also not supported.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"]
    }

    def __init__(self, const_throttle=None):
        super(AutoDriftEnv, self).__init__()
        self.const_throttle = const_throttle

        # save the last n commands (throttle and steering)
        self.n_commands = 2

        # interface to the car
        self.viewer = CarController()

        # action space
        if const_throttle is None:
            # define independent bounds for steering and throttle
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]),
                                           high=np.array([MAX_STEERING, 1]), dtype=np.float32)
        else:
            # steering only
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING]),
                                           high=np.array([MAX_STEERING]), dtype=np.float32)

        # use pixels as input
        self.observation_space = spaces.Box(low=0, high=255, shape=INPUT_DIM, dtype=np.uint8)

    def step(self, action):
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict
        """
        # action[0] = steering angle, action[1] = throttle
        if self.const_throttle is None:
            # variable throttle. convert throttle range -> [-1, 1] -> [0, 1] -> [MIN, MAX]
            temp = (action[1] + 1) / 2
            action[1] = (1 - temp) * MIN_THROTTLE + temp * MAX_THROTTLE
        else:
            # fixed throttle
            action = np.concatenate([action, [self.const_throttle]])

        # order the car
        self.viewer.take_action(action)
        observation, reward, done, info = self.viewer.observe()
        logger.debug("Reward: {0}, isDone: {1}".format(reward, done))

        return observation, reward, done, info

    def reset(self):
        print("One env timestep done. Resetting after 0.5s...")
        # reset state of car controller, then sleep for 5s to replace car to start position manually
        self.viewer.reset()
        time.sleep(0.5)
        print("Now resuming...")

        observation, reward, done, info = self.viewer.observe()

        return observation

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.viewer.image_array
        return None

    def close(self):
        # nothing actually happens here
        self.viewer.quit()