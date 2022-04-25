from collections import deque
import torch
import torch.multiprocessing as mp
import copy
import numpy as np
import gym
from torchrl.env.vecenv import VecEnv


class BaseCollector:
  def __init__(
      self,
      env, eval_env, pf, replay_buffer,
      epoch_frames,
      train_render=False,
      eval_episodes=1,
      eval_render=False,
      device='cpu',
      max_episode_frames=999):

    self.pf = pf
    self.replay_buffer = replay_buffer

    self.env = env
    self.env.train()
    self.continuous = isinstance(self.env.action_space, gym.spaces.Box)
    self.train_render = train_render

    if eval_env is not None:
      self.eval_env = eval_env
    else:
      self.eval_env = copy.deepcopy(env)
      if hasattr(env, "_obs_normalizer"):
        self.eval_env._obs_normalizer = env._obs_normalizer
    self.eval_env._reward_scale = 1
    self.eval_episodes = eval_episodes
    self.eval_render = eval_render

    self.current_ob = self.env.reset()

    self.train_rew = 0

    # device specification
    self.device = device

    self.to(self.device)

    self.epoch_frames = epoch_frames
    self.sample_epoch_frames = epoch_frames
    self.max_episode_frames = max_episode_frames

    self.current_step = 0

  def start_episode(self):
    pass

  def finish_episode(self):
    pass

  def take_actions(self):
    out = self.pf.explore(
      torch.Tensor(self.current_ob).to(self.device).unsqueeze(0))
    act = out["action"]
    act = act.detach().cpu().numpy()

    if not self.continuous:
      act = act[0]
    elif np.isnan(act).any():
      print("NaN detected. BOOM")
      exit()

    next_ob, reward, done, info = self.env.step(act)
    if self.train_render:
      self.env.render()
    self.current_step += 1
    self.train_rew += reward
    sample_dict = {
      "obs": np.expand_dims(self.current_ob, 0),
      "next_obs": np.expand_dims(next_ob, 0),
      "acts": np.expand_dims(act, 0),
      "rewards": [[reward]],
      "terminals": [[done]],
      "time_limits": [[
        info["time_limit"] if "time_limit" in info else False]]
    }

    if done or self.current_step >= self.max_episode_frames:
      # if done:
      next_ob = self.env.reset()
      self.finish_episode()
      self.start_episode()
      # reset current_step
      self.current_step = 0

      # self.training_episode_rewards.append(self.train_rew)
      self.train_rews.append(self.train_rew)
      self.train_rew = 0

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_ob

    return reward

  def terminate(self):
    self.env.close()
    self.eval_env.close()

  def train_one_epoch(self):
    self.train_rews = []
    self.train_epoch_reward = 0
    self.env.train()

    for _ in range(self.sample_epoch_frames):
      # Sample actions
      reward = self.take_actions()

      self.train_epoch_reward += reward

    return {
      'train_rewards': self.train_rews,
      'train_epoch_reward': self.train_epoch_reward
    }

  def eval_one_epoch(self):
    eval_infos = {}
    eval_rews = []

    done = False
    if hasattr(self.env, "_obs_normalizer"):
      self.eval_env._obs_normalizer = copy.deepcopy(self.env._obs_normalizer)
    self.eval_env.eval()

    traj_lens = []
    for _ in range(self.eval_episodes):

      eval_ob = self.eval_env.reset()
      rew = 0
      traj_len = 0
      done = False
      while not done:
        act = self.pf.eval_act(torch.Tensor(eval_ob).to(
          self.device).unsqueeze(0))

        if self.continuous and np.isnan(act).any():
          print("NaN detected. BOOM")
          exit()
        try:
          eval_ob, r, done, _ = self.eval_env.step(act)
          rew += r
          traj_len += 1
          if self.eval_render:
            self.eval_env.render()
        except Exception:
          print(act)

      eval_rews.append(rew)
      traj_lens.append(traj_len)

      done = False

    eval_infos["eval_rewards"] = eval_rews
    eval_infos["eval_traj_length"] = np.mean(traj_lens)
    return eval_infos

  def to(self, device):
    for func in self.funcs:
      self.funcs[func].to(device)

  @property
  def funcs(self):
    return {
      "pf": self.pf
    }


class VecCollector(BaseCollector):
  def __init__(self, **kwargs):
    super(VecCollector, self).__init__(**kwargs)
    self.sample_epoch_frames //= self.env.env_nums
    # assert isinstance(self.env, VecEnv)
    self.current_step = np.zeros((self.env.env_nums, 1))
    self.train_rew = np.zeros_like(self.current_step)

  def take_actions(self):
    out = self.pf.explore(
      torch.Tensor(self.current_ob).to(self.device).unsqueeze(0))
    act = out["action"]
    act = act.detach().cpu().numpy()

    if not self.continuous:
      act = act[..., 0]
    elif np.isnan(act).any():
      print("NaN detected. BOOM")
      print(self.pf.forward(
        torch.Tensor(self.current_ob).to(self.device)
      ))
      exit()

    next_ob, reward, done, infos = self.env.step(act)
    if self.train_render:
      self.env.render()
    self.current_step += 1

    sample_dict = {
      "obs": self.current_ob,
      "next_obs": next_ob,
      "acts": act,
      "rewards": reward,
      "terminals": done,
      "time_limits":
      infos["time_limit"][:, np.newaxis]
      if "time_limit" in infos else [False]
    }

    self.train_rew += reward
    if np.any(done):
      self.train_rews += list(self.train_rew[done])
      self.train_rew[done] = 0

    if np.any(done) or \
       np.any(self.current_step >= self.max_episode_frames):
      # if np.any(done):
      #     flag = done
      flag = (self.current_step >= self.max_episode_frames) | done
      next_ob = self.env.partial_reset(np.squeeze(flag, axis=-1))
      self.current_step[flag] = 0

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_ob

    return np.sum(reward)

  def eval_one_epoch(self):
    eval_infos = {}
    eval_rews = []

    print(self.env._obs_normalizer._mean)
    if hasattr(self.env, "_obs_normalizer"):
      self.eval_env._obs_normalizer = copy.deepcopy(self.env._obs_normalizer)

    self.eval_env.eval()

    print(self.env._obs_normalizer._mean)
    print(self.eval_env._obs_normalizer._mean)

    traj_lens = []
    for _ in range(self.eval_episodes):
      done = np.zeros((self.eval_env.env_nums, 1)).astype(np.bool)
      epi_done = np.zeros((self.eval_env.env_nums, 1)).astype(np.bool)

      eval_obs = self.eval_env.reset()

      rews = np.zeros_like(done)
      traj_len = np.zeros_like(rews)

      while not np.all(epi_done):
        act = self.pf.eval_act(
          torch.Tensor(eval_obs).to(self.device)
        )
        if self.continuous and np.isnan(act).any():
          print("NaN detected. BOOM")
          print(self.pf.forward(torch.Tensor(eval_obs).to(self.device)))
          exit()
        try:
          eval_obs, r, done, _ = self.eval_env.step(act)
          rews = rews + ((1-epi_done) * r)
          traj_len = traj_len + (1 - epi_done)

          epi_done = epi_done | done
          if np.any(done):
            eval_obs = self.eval_env.partial_reset(
              np.squeeze(done, axis=-1)
            )

          if self.eval_render:
            self.eval_env.render()
        except Exception as e:
          print(e)
          print(act)
          exit()
      eval_rews += list(rews)
      traj_lens += list(traj_len)

    eval_infos["eval_rewards"] = eval_rews
    eval_infos["eval_traj_length"] = np.mean(traj_lens)
    return eval_infos
