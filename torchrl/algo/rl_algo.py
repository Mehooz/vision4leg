import copy
import time
from collections import deque
import numpy as np
import torch
import torchrl.algo.utils as atu
import gym
import os
import os.path as osp
import pathlib
import pickle


class RLAlgo():
  """
  Base RL Algorithm Framework
  """

  def __init__(
      self,
      env=None,
      replay_buffer=None,
      collector=None,
      logger=None,
      grad_clip=None,
      discount=0.99,
      num_epochs=3000,
      batch_size=128,
      device='cpu',
      save_interval=100,
      eval_interval=1,
      save_dir=None):

    self.env = env

    self.continuous = isinstance(self.env.action_space, gym.spaces.Box)

    self.replay_buffer = replay_buffer
    self.collector = collector
    # device specification
    self.device = device

    # environment relevant information
    self.discount = discount
    self.num_epochs = num_epochs
    self.epoch_frames = self.collector.epoch_frames

    # training information
    self.batch_size = batch_size
    self.training_update_num = 0
    self.sample_key = None
    self.grad_clip = grad_clip

    # Logger & relevant setting
    self.logger = logger

    self.episode_rewards = deque(maxlen=30)
    self.training_episode_rewards = deque(maxlen=30)

    self.save_interval = save_interval
    self.save_dir = save_dir

    pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    self.best_eval = None
    self.eval_interval = eval_interval

    self.explore_time = 0
    self.train_time = 0
    self.start = time.time()

  def start_epoch(self):
    pass

  def finish_epoch(self):
    return {}

  def pretrain(self):
    pass

  def update_per_epoch(self):
    pass

  def snapshot(self, prefix, epoch):
    if hasattr(self.env, "_obs_normalizer") and \
        self.env._obs_normalizer is not None:
      normalizer_file_name = "_obs_normalizer_{}.pkl".format(epoch)
      normalizer_path = osp.join(prefix, normalizer_file_name)
      with open(normalizer_path, "wb") as f:
        pickle.dump(self.env._obs_normalizer, f)

    for name, network in self.snapshot_networks:
      model_file_name = "model_{}_{}.pth".format(name, epoch)
      model_path = osp.join(prefix, model_file_name)
      torch.save(network.state_dict(), model_path)

  def train(self):
    self.pretrain()
    total_frames = 0
    if hasattr(self, "pretrain_frames"):
      total_frames = self.pretrain_frames

    self.start_epoch()

    for epoch in range(self.num_epochs):
      self.current_epoch = epoch
      start = time.time()

      self.start_epoch()

      explore_start_time = time.time()
      training_epoch_info = self.collector.train_one_epoch()
      for reward in training_epoch_info["train_rewards"]:
        self.training_episode_rewards.append(reward)

      self.explore_time += time.time() - explore_start_time

      train_start_time = time.time()
      self.update_per_epoch()
      self.train_time += time.time() - train_start_time

      finish_epoch_info = self.finish_epoch()

      total_frames += self.epoch_frames

      if epoch % self.eval_interval == 0:
        eval_start_time = time.time()
        eval_infos = self.collector.eval_one_epoch()
        eval_time = time.time() - eval_start_time

        infos = {}

        for reward in eval_infos["eval_rewards"]:
          self.episode_rewards.append(reward)

        if self.best_eval is None or \
            (np.mean(eval_infos["eval_rewards"]) > self.best_eval):
          self.best_eval = np.mean(eval_infos["eval_rewards"])
          self.snapshot(self.save_dir, 'best')
          print("Best Saved: {:.5f},  EPoch: {}".format(
            np.mean(eval_infos["eval_rewards"]),
            epoch
          ))
        del eval_infos["eval_rewards"]

        infos["Running_Average_Rewards"] = np.mean(
          self.episode_rewards)
        infos["Train_Epoch_Reward"] = \
          training_epoch_info["train_epoch_reward"]
        infos["Running_Training_Average_Rewards"] = np.mean(
          self.training_episode_rewards)
        infos["Explore_Time"] = self.explore_time
        infos["Train___Time"] = self.train_time
        infos["Eval____Time"] = eval_time
        self.explore_time = 0
        self.train_time = 0
        infos.update(eval_infos)
        infos.update(finish_epoch_info)

        self.logger.add_epoch_info(
          epoch, total_frames, time.time() - self.start, infos)
        self.start = time.time()

      if epoch % self.save_interval == 0:
        self.snapshot(self.save_dir, epoch)

    self.snapshot(self.save_dir, "finish")
    self.collector.terminate()

  def update(self, batch):
    raise NotImplementedError

  def _update_target_networks(self):
    if self.use_soft_update:
      for net, target_net in self.target_networks:
        atu.soft_update_from_to(net, target_net, self.tau)
    else:
      if self.training_update_num % self.target_hard_update_period == 0:
        for net, target_net in self.target_networks:
          atu.copy_model_params_from_to(net, target_net)

  @property
  def networks(self):
    return [
    ]

  @property
  def snapshot_networks(self):
    return [
    ]

  @property
  def target_networks(self):
    return [
    ]

  def to(self, device):
    for net in self.networks:
      net.to(device)
