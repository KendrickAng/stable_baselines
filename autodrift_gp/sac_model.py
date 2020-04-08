import time
from collections import deque

import numpy as np
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.common import TensorboardWriter, SetVerbosity

import logging
from logging import INFO, DEBUG
logging.basicConfig(level=DEBUG, format="%(levelname)s [line %(lineno)d]: %(message)s")
logger = logging.getLogger()
logger.disabled = False


class SacModel(SAC):
    """
    Custom version of the Soft-Actor Critic (SAC) adapted from Stable-Baselines.
    Original: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/sac.py

    Notable changes:
    - optimization is done after each episode and not when num_timestep % train_freq == 0 in the original
    - this version is integrated with teleoperation
    """
    def optimize(self, step, writer, current_lr):
        """
        Several additional optimization steps when updating the different networks.

        :param step: (int) current time-step
        :param writer: (TensorboardWriter object)
        :param current_lr: (float) current learning rate
        :return: ([np.ndarray] values used for monitoring
        """
        optim_start = time.time()
        mb_infos_vals = []
        for grad_step in range(self.gradient_steps):
            # not enough data to optimise yet
            if step < self.batch_size or step < self.learning_starts:
                break

            # Update policy and critics (q-functions) and extra logging
            # _train_step: returns (policy_loss, qf1_loss, qf2_loss, value_loss, entropy)
            mb_infos_vals.append(self._train_step(step, writer, current_lr))
            self.n_updates += 1

            if (step + grad_step) % self.target_update_interval == 0:
                # Replace weights of the separate target network
                self.sess.run(self.target_update_op)

        if self.n_updates > 0:
            print("SAC training duration: {:.2f}".format(time.time() - optim_start))

        return mb_infos_vals

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):
        print_freq = 100

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)

            start_time = time.time()
            episode_rewards = [0.0]

            # Todo: Extra
            # is_teleop_env = hasattr(self.env, "wait_for_teleop_reset")
            # # TeleopEnv
            # if is_teleop_env:
            #     print("Waiting for teleop")
            #     obs = self.env.wait_for_teleop_reset()
            # # Todo: End extra
            # else:
            obs = self.env.reset()

            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            ep_len = 0
            self.n_updates = 0
            infos_values = []
            mb_infos_vals = []

            for step in range(total_timesteps):
                logger.info("timestep {0}".format(step))
                # Compute current learning_rate
                frac = 1.0 - step / total_timesteps
                current_lr = self.learning_rate(frac)

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if step < self.learning_starts:
                    action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                ep_len += 1

                if print_freq > 0 and ep_len % print_freq == 0 and ep_len > 0:
                    print("{} steps".format(ep_len))

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, step)

                # TODO: Extra
                if ep_len > self.train_freq:
                    print("Additional training")
                    self.env.reset()
                    mb_infos_vals = self.optimize(step, writer, current_lr)
                    done = True
                # TODO: End extra

                episode_rewards[-1] += reward
                if done:
                    # if not (isinstance(self.env, VecEnv) or is_teleop_env):
                    #     obs = self.env.reset()
                    if not (isinstance(self.env, VecEnv)):
                        obs = self.env.reset()

                    print("Episode finished. Reward: {:.2f} {} Steps".format(episode_rewards[-1], ep_len))
                    episode_rewards.append(0.0)
                    ep_len = 0
                    mb_infos_vals = self.optimize(step, writer, current_lr)

                    # Todo: Extra
                    # Refresh obs when using TeleopEnv
                    # if is_teleop_env:
                    #     print("Waiting for teleop")
                    #     obs = self.env.wait_for_teleop_reset()
                    # Todo: End extra

                # Log losses and entropy, useful for monitor training
                if len(mb_infos_vals) > 0:
                    infos_values = np.mean(mb_infos_vals, axis=0)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", self.n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', "{:.2f}".format(time.time() - start_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", step)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []

            # TODO: Extra
            # if is_teleop_env:
            #     self.env.is_training = False
            # Use last batch
            print("Final optimization before saving")
            self.env.reset()
            mb_infos_vals = self.optimize(step, writer, current_lr)
            # TODO: End extra
        return self