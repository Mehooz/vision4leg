import gym
import numpy as np
import copy
import torch


class BaseWrapper(gym.Wrapper):
  def __init__(self, env):
    super(BaseWrapper, self).__init__(env)
    self._wrapped_env = env
    self.training = True

  def train(self):
    if isinstance(self._wrapped_env, BaseWrapper):
      self._wrapped_env.train()
    self.training = True

  def eval(self):
    if isinstance(self._wrapped_env, BaseWrapper):
      self._wrapped_env.eval()
    self.training = False

  def __getattr__(self, attr):
    if attr == '_wrapped_env':
      raise AttributeError()
    return getattr(self._wrapped_env, attr)

  def copy_state(self, source_env):
    pass


class RewardShift(gym.RewardWrapper, BaseWrapper):
  def __init__(self, env, reward_scale=1):
    super(RewardShift, self).__init__(env)
    self._reward_scale = reward_scale

  def reward(self, reward):
    if self.training:
      return self._reward_scale * reward
    else:
      return reward


def update_mean_var_count(
    mean, var, count,
    batch_mean, batch_var, batch_count):
  """
  Imported From OpenAI Baseline
  """
  delta = batch_mean - mean
  tot_count = count + batch_count

  new_mean = mean + delta * batch_count / tot_count
  m_a = var * count
  m_b = batch_var * batch_count
  M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
  new_var = M2 / tot_count
  new_count = tot_count

  return new_mean, new_var, new_count


class Normalizer():
  def __init__(self, shape, clip=10.):
    self.shape = shape
    self._mean = np.zeros(shape)
    self._var = np.ones(shape)
    self._count = 1e-4
    self.clip = clip
    self.should_estimate = True

  def stop_update_estimate(self):
    self.should_estimate = False

  def update_estimate(self, data):
    if not self.should_estimate:
      return
    if len(data.shape) == self.shape:
      data = data[np.newaxis, :]
    self._mean, self._var, self._count = update_mean_var_count(
      self._mean, self._var, self._count,
      np.mean(data, axis=0), np.var(data, axis=0), data.shape[0])

  def inverse(self, raw):
    return raw * np.sqrt(self._var) + self._mean

  def inverse_torch(self, raw):
    return raw * torch.Tensor(np.sqrt(self._var)).to(raw.device) \
      + torch.Tensor(self._mean).to(raw.device)

  def filt(self, raw):
    return np.clip(
      (raw - self._mean) / (np.sqrt(self._var) + 1e-4),
      -self.clip, self.clip)

  def filt_torch(self, raw):
    return torch.clamp(
      (raw - torch.Tensor(self._mean).to(raw.device)) /
      (torch.Tensor(np.sqrt(self._var) + 1e-4).to(raw.device)),
      -self.clip, self.clip)


class NormObs(gym.ObservationWrapper, BaseWrapper):
  """
  Normalized Observation => Optional, Use Momentum
  """

  def __init__(self, env, epsilon=1e-4, clipob=10.):
    super(NormObs, self).__init__(env)
    self.count = epsilon
    self.clipob = clipob
    self._obs_normalizer = Normalizer(env.observation_space.shape)

  def copy_state(self, source_env):
    # self._obs_rms = copy.deepcopy(source_env._obs_rms)
    self._obs_var = copy.deepcopy(source_env._obs_var)
    self._obs_mean = copy.deepcopy(source_env._obs_mean)

  def observation(self, observation):
    if self.training:
      self._obs_normalizer.update_estimate(observation)
    return self._obs_normalizer.filt(observation)


class NormRet(BaseWrapper):
  def __init__(self, env, discount=0.99, epsilon=1e-4):
    super(NormRet, self).__init__(env)
    self._ret = 0
    self.count = 1e-4
    self.ret_mean = 0
    self.ret_var = 1
    self.discount = discount
    self.epsilon = 1e-4

  def step(self, act):
    obs, rews, done, infos = self.env.step(act)
    if self.training:
      self.ret = self.ret * self.discount + rews
      # if self.ret_rms:
      self.ret_mean, self.ret_var, self.count = update_mean_var_count(
        self.ret_mean, self.ret_var, self.count, self.ret, 0, 1)
      rews = rews / np.sqrt(self.ret_var + self.epsilon)
      self.ret *= (1-done)
    return obs, rews, done, infos

  def reset(self, **kwargs):
    self.ret = 0
    return self.env.reset(**kwargs)


# Check Trajectory is ended by time limit or not
class TimeLimitAugment(BaseWrapper):
  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    info['time_limit'] = done \
      and self.env._max_episode_steps == self.env._elapsed_steps
    return obs, rew, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)
