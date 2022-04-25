import numpy as np
from .vecenv import VecEnv
import multiprocessing as mp
from toolz.dicttoolz import merge_with
import sys

mp.set_start_method('spawn', force=True)

mp.set_start_method('spawn', force=True)


def env_worker(
    env_funcs, env_args, child_pipe, parent_pipe
):
  envs = [
    env_func(*env_arg)
    for env_func, env_arg in zip(env_funcs, env_args)
  ]
  sys.stderr = open("train_process.stderr", "a")
  sys.stdout = open("train_process.stdout", "a")

  # parent_pipe.close()

  try:
    while True:
      command, data = child_pipe.recv()
      if command == 'step':
        results = [
          env.step(np.squeeze(action)) for env, action in zip(envs, data)
        ]
        child_pipe.send(results)
      elif command == 'reset':
        results = [env.reset(**data) for env in envs]
        child_pipe.send(results)
      elif command == 'partial_reset':
        index_mask, kwargs = data
        indexs = np.argwhere(index_mask == 1).reshape((-1))
        results = [envs[index].reset(**kwargs) for index in indexs]
        child_pipe.send(results)
      # elif command == 'render':
      #     child_pipe.send(env.render(mode='rgb_array'))
      elif command == 'train':
        for env in envs:
          env.train()
      elif command == 'eval':
        for env in envs:
          env.eval()
      elif command == 'close':
        child_pipe.close()
        break
  except Exception as e:
    print(e)
  finally:
    for env in envs:
      env.close()


class SubProcVecEnv(VecEnv):
  def __init__(self, proc_nums, env_nums, env_funcs, env_args):
    self.proc_nums = proc_nums
    super().__init__(env_nums, env_funcs, env_args)

  def set_up_envs(self):
    self.example_env = self.env_funcs[0](*self.env_args[0])
    self.workers = []
    self.parent_pipes = []

    assert self.env_nums % self.proc_nums == 0
    self.env_nums_per_proc = self.env_nums // self.proc_nums

    self.ctx = mp.get_context()
    for i in range(self.proc_nums):
      env_idx_start = i * self.env_nums_per_proc
      env_idx_end = (i + 1) * self.env_nums_per_proc
      parent_pipe, child_pipe = self.ctx.Pipe()
      p = self.ctx.Process(
        target=env_worker,
        args=(
          self.env_funcs[env_idx_start: env_idx_end],
          self.env_args[env_idx_start: env_idx_end],
          child_pipe,
          parent_pipe
        )
      )
      p.start()
      # child_pipe.close()
      self.workers.append(p)
      self.parent_pipes.append(parent_pipe)

  def train(self):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('train', None))

  def eval(self):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('eval', None))

  def close(self):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('close', None))

  def reset(self, **kwargs):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('reset', kwargs))

    obs = []
    for parent_pipe in self.parent_pipes:
      obs += parent_pipe.recv()

    self._obs = np.stack(obs)
    return self._obs

  def partial_reset(self, index_mask, **kwargs):
    index_mask_per_proc = np.split(index_mask, self.proc_nums)
    for index_mask_current, parent_pipe in zip(
        index_mask_per_proc, self.parent_pipes):
      parent_pipe.send(
        ('partial_reset', (index_mask_current, kwargs))
      )

    partial_obs = []
    for parent_pipe in self.parent_pipes:
      partial_obs += parent_pipe.recv()
    self._obs[index_mask] = partial_obs
    return self._obs

  def step(self, actions):
    actions = np.split(actions, self.proc_nums * self.env_nums_per_proc)
    for index, parent_pipe in enumerate(self.parent_pipes):
      parent_pipe.send((
        'step',
        actions[
          index * self.env_nums_per_proc: (index + 1) * self.env_nums_per_proc
        ]
      ))
    results = []
    for parent_pipe in self.parent_pipes:
      results += parent_pipe.recv()

    obs, rews, dones, infos = zip(*results)
    self._obs = np.stack(obs)
    infos = merge_with(np.array, *infos)
    return self._obs, np.stack(rews)[:, np.newaxis], \
      np.stack(dones)[:, np.newaxis], infos

  def seed(self, seed):
    for idx, parent_pipe in enumerate(self.parent_pipes):
      parent_pipe.send(('seed', seed * self.env_nums + idx))
    # for idx, env in enumerate(self.envs):
    #     env.seed(seed * self.env_nums + idx)

  @property
  def observation_space(self):
    return self.example_env.observation_space

  @property
  def action_space(self):
    return self.example_env.action_space

  def __getattr__(self, attr):
    if attr == '_wrapped_env':
      raise AttributeError()
    return getattr(self.example_env, attr)
