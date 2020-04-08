# load the pretrained weights and start the model and stuff
from gym_env import AutoDriftEnv
from stable_baselines.sac.policies import MlpPolicy
from sac_model import SacModel
from skimage import io
from skimage.transform import resize

N_TIMESTEPS = 2000 # enjoy agent for 2000 timesteps (a timestep is a state-action-state transition)
LOAD_PATH = './'

if __name__ == '__main__':
    env = AutoDriftEnv(const_throttle=None)
    model = SacModel(policy=MlpPolicy, env=env)

    # try to predict once with (80, 102, 3) image
    img = io.imread('picture.jpg')
    img_resized = resize(img, (80, 160, 3))
    print(img_resized.shape)

    action, _ = model.predict(observation=img_resized)
    print(action)
    #
    # action, _states = model.predict(img)
    #
    # print(action)
    # print(_states)

    # # Force deterministic for SAC
    # deterministic = True
    # running_reward = 0.0
    # ep_len = 0
    # for _ in range(N_TIMESTEPS):
    #     pass
    #
    # pass
