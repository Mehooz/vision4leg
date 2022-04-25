import numpy as np
from .base_wrapper import BaseWrapper
from toolz.dicttoolz import merge_with


class VecEnv(BaseWrapper):
  """
  Vector Env
      Each env should have
      1. same observation space shape
      2. same action space shape
  """

  def __init__(self, env_nums, env_funcs, env_args):
    self.env_nums = env_nums
    self.env_funcs = env_funcs
    self.env_args = env_args
    if isinstance(env_funcs, list):
      assert len(env_funcs) == env_nums
      assert len(env_args) == env_nums
    else:
      self.env_funcs = [env_funcs for _ in range(env_nums)]
      self.env_args = [env_args for _ in range(env_nums)]

    self.set_up_envs()

  def set_up_envs(self):
    self.envs = [env_func(*env_arg) for env_func, env_arg
                 in zip(self.env_funcs, self.env_args)]

  def train(self):
    for env in self.envs:
      env.train()

  def eval(self):
    for env in self.envs:
      env.eval()

  def close(self):
    for env in self.envs:
      env.close()

  def reset(self, **kwargs):
    obs = [env.reset() for env in self.envs]
    self._obs = np.stack(obs)
    return self._obs

  def partial_reset(self, index_mask, **kwargs):
    indexs = np.argwhere(index_mask == 1).reshape((-1))
    reset_obs = [self.envs[index].reset() for index in indexs]
    self._obs[index_mask] = reset_obs
    return self._obs

  def step(self, actions):
    actions = np.split(actions, self.env_nums)
    result = [env.step(np.squeeze(action)) for env, action in
              zip(self.envs, actions)]
    obs, rews, dones, infos = zip(*result)
    self._obs = np.stack(obs)
    infos = merge_with(np.array, *infos)
    return self._obs, np.stack(rews)[:, np.newaxis], \
      np.stack(dones)[:, np.newaxis], infos

  def seed(self, seed):
    # for env in self.envs:
    #     env.seed(seed)
    for idx, env in enumerate(self.envs):
      env.seed(seed * self.env_nums + idx)

  @property
  def observation_space(self):
    return self.envs[0].observation_space

  @property
  def action_space(self):
    return self.envs[0].action_space

  def __getattr__(self, attr):
    if attr == '_wrapped_env':
      raise AttributeError()
    return getattr(self.envs[0], attr)
