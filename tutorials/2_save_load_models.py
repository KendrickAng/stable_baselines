"""
This tutorial covers saving and loading of models.
"""
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

# Check for GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# use loaded model?
USE_LOADED_MODEL = False

env = gym.make('CartPole-v1')

def main():
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor

    if not USE_LOADED_MODEL:
        model = PPO2(MlpPolicy, env, verbose=1)

        # before training
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print("Mean reward: {0}, Std reward: {1}".format(mean_reward, std_reward))

        model.learn(total_timesteps=5000)

        # save model
        model.save("cartpole_v1_ppo2")

    loaded_model = PPO2.load("cartpole_v1_ppo2")
    loaded_model.set_env(env)

    # after training
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=50)
    print("Mean reward: {0} +/- {1}".format(mean_reward, std_reward))

    # Visualise trained agent
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == '__main__':
    main()