"""
This tutorial covers using multiprocessing to speed up training.
Each process runs an independent instance of the Gym environment.
It seems that SAC cannot be parallelised.
"""
import gym
import time
from stable_baselines import SAC
from stable_baselines import ACKTR
from stable_baselines.sac import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.evaluation import evaluate_policy

USE_LOADED_MODEL = False

# env_id = 'Pendulum-v0'
env_id = 'CartPole-v1'
num_cpu = 4
n_timesteps = 25000

def main():
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    if not USE_LOADED_MODEL:
        model = ACKTR('MlpPolicy', env, verbose=1)

        # Multiprocessed RL Training
        start_time = time.time()
        model.learn(total_timesteps=n_timesteps, log_interval=10)
        total_time_multi = time.time() - start_time

        model.save("cartpole_v1_acktr")

    loaded_model = ACKTR.load("cartpole_v1_acktr")
    loaded_model.set_env(env)

    # Single Process RL Training
    single_process_model = ACKTR('MlpPolicy', env_id, verbose=1)
    start_time = time.time()
    single_process_model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print("Single-process: {0}s, Multi-process: {1}s".format(total_time_single, total_time_multi))

    # create separate clean environment for evaluation
    eval_env = gym.make(env_id)
    mean_reward, std_reward = evaluate_policy(loaded_model, eval_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init

if __name__ == "__main__":
    main()