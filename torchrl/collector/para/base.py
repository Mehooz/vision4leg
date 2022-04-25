
import torch
import torch.multiprocessing as mp
import copy
import numpy as np
import gym
from collections import deque

from torchrl.collector.base import BaseCollector
from torchrl.collector.base import EnvInfo

from torchrl.replay_buffers.shared import SharedBaseReplayBuffer

TIMEOUT_CHILD = 200


class ParallelCollector(BaseCollector):
  def __init__(
    self,
    env, pf, replay_buffer,
    env_cls, env_args,
    train_epochs,
    eval_epochs,
    worker_nums=4,
    eval_worker_nums=1,
      **kwargs):

    super().__init__(
      env, pf, replay_buffer,
      **kwargs)

    self.env_cls = env_cls
    self.env_args = env_args

    self.env_info.device = 'cpu'  # CPU For multiprocess sampling
    self.shared_funcs = copy.deepcopy(self.funcs)
    for key in self.shared_funcs:
      self.shared_funcs[key].to(self.env_info.device)

    assert isinstance(replay_buffer, SharedBaseReplayBuffer), \
      "Should Use Shared Replay buffer"
    self.replay_buffer = replay_buffer

    self.worker_nums = worker_nums
    self.eval_worker_nums = eval_worker_nums

    self.manager = mp.Manager()
    self.train_epochs = train_epochs
    self.eval_epochs = eval_epochs
    self.start_worker()

  @staticmethod
  def train_worker_process(cls, shared_funcs, env_info,
                           replay_buffer, shared_que,
                           start_barrier, epochs):

    replay_buffer.rebuild_from_tag()
    local_funcs = copy.deepcopy(shared_funcs)
    for key in local_funcs:
      local_funcs[key].to(env_info.device)

    # Rebuild Env
    env_info.env = env_info.env_cls(**env_info.env_args)

    c_ob = {
      "ob": env_info.env.reset()
    }
    train_rew = 0
    current_epoch = 0
    while True:
      start_barrier.wait()
      current_epoch += 1
      if current_epoch > epochs:
        break

      for key in shared_funcs:
        local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

      train_rews = []
      train_epoch_reward = 0

      for _ in range(env_info.epoch_frames):
        next_ob, done, reward, _ = cls.take_actions(
          local_funcs, env_info, c_ob, replay_buffer)
        c_ob["ob"] = next_ob
        train_rew += reward
        train_epoch_reward += reward
        if done:
          train_rews.append(train_rew)
          train_rew = 0

      shared_que.put({
        'train_rewards': train_rews,
        'train_epoch_reward': train_epoch_reward
      })

  @staticmethod
  def eval_worker_process(shared_pf,
                          env_info, shared_que, start_barrier, epochs):

    pf = copy.deepcopy(shared_pf).to(env_info.device)

    # Rebuild Env
    env_info.env = env_info.env_cls(**env_info.env_args)

    env_info.env.eval()
    env_info.env._reward_scale = 1
    current_epoch = 0

    while True:
      start_barrier.wait()
      current_epoch += 1
      if current_epoch > epochs:
        break
      pf.load_state_dict(shared_pf.state_dict())

      eval_rews = []

      done = False
      for _ in range(env_info.eval_episodes):

        eval_ob = env_info.env.reset()
        rew = 0
        while not done:
          act = pf.eval_act(torch.Tensor(eval_ob).to(
            env_info.device).unsqueeze(0))
          eval_ob, r, done, _ = env_info.env.step(act)
          rew += r
          if env_info.eval_render:
            env_info.env.render()

        eval_rews.append(rew)
        done = False

      shared_que.put({
        'eval_rewards': eval_rews
      })

  def start_worker(self):
    self.workers = []
    self.shared_que = self.manager.Queue(self.worker_nums)
    self.start_barrier = mp.Barrier(self.worker_nums+1)

    self.eval_workers = []
    self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
    self.eval_start_barrier = mp.Barrier(self.eval_worker_nums+1)

    self.env_info.env_cls = self.env_cls
    self.env_info.env_args = self.env_args

    for i in range(self.worker_nums):
      self.env_info.env_rank = i
      p = mp.Process(
        target=self.__class__.train_worker_process,
        args=(self.__class__, self.shared_funcs,
              self.env_info, self.replay_buffer,
              self.shared_que, self.start_barrier,
              self.train_epochs))
      p.start()
      self.workers.append(p)

    for i in range(self.eval_worker_nums):
      eval_p = mp.Process(
        target=self.__class__.eval_worker_process,
        args=(self.shared_funcs["pf"],
              self.env_info, self.eval_shared_que, self.eval_start_barrier,
              self.eval_epochs))
      eval_p.start()
      self.eval_workers.append(eval_p)

  def terminate(self):
    self.start_barrier.wait()
    self.eval_start_barrier.wait()
    for p in self.workers:
      p.join()

    for p in self.eval_workers:
      p.join()

  def train_one_epoch(self):
    self.start_barrier.wait()
    train_rews = []
    train_epoch_reward = 0

    for key in self.shared_funcs:
      self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
    for _ in range(self.worker_nums):
      worker_rst = self.shared_que.get()
      train_rews += worker_rst["train_rewards"]
      train_epoch_reward += worker_rst["train_epoch_reward"]

    return {
      'train_rewards': train_rews,
      'train_epoch_reward': train_epoch_reward
    }

  def eval_one_epoch(self):
    self.eval_start_barrier.wait()
    eval_rews = []

    self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

    for _ in range(self.eval_worker_nums):
      worker_rst = self.eval_shared_que.get()
      eval_rews += worker_rst["eval_rewards"]

    return {
      'eval_rewards': eval_rews,
    }

  @property
  def funcs(self):
    return {
      "pf": self.pf
    }


class AsyncParallelCollector(ParallelCollector):
  def start_worker(self):
    self.workers = []
    self.shared_que = self.manager.Queue(self.worker_nums)
    self.start_barrier = mp.Barrier(self.worker_nums)

    self.eval_workers = []
    self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
    self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

    self.env_info.env_cls = self.env_cls
    self.env_info.env_args = self.env_args

    for i in range(self.worker_nums):
      self.env_info.env_rank = i
      p = mp.Process(
        target=self.__class__.train_worker_process,
        args=(self.__class__, self.shared_funcs,
              self.env_info, self.replay_buffer,
              self.shared_que, self.start_barrier,
              self.train_epochs))
      p.start()
      self.workers.append(p)

    for i in range(self.eval_worker_nums):
      eval_p = mp.Process(
        target=self.__class__.eval_worker_process,
        args=(self.pf,
              self.env_info, self.eval_shared_que, self.eval_start_barrier,
              self.eval_epochs))
      eval_p.start()
      self.eval_workers.append(eval_p)

  def terminate(self):
    # self.eval_start_barrier.wait()
    for p in self.workers:
      p.join()

    for p in self.eval_workers:
      p.join()

  def train_one_epoch(self):
    train_rews = []
    train_epoch_reward = 0

    for key in self.shared_funcs:
      self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
    for _ in range(self.worker_nums):
      worker_rst = self.shared_que.get()
      train_rews += worker_rst["train_rewards"]
      train_epoch_reward += worker_rst["train_epoch_reward"]

    return {
      'train_rewards': train_rews,
      'train_epoch_reward': train_epoch_reward
    }

  def eval_one_epoch(self):
    # self.eval_start_barrier.wait()
    eval_rews = []
    self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())
    for _ in range(self.eval_worker_nums):
      worker_rst = self.eval_shared_que.get()
      eval_rews += worker_rst["eval_rewards"]

    return {
      'eval_rewards': eval_rews,
    }
