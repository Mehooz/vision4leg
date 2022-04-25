import torch
import numpy as np
import copy
from .base import BaseCollector, VecCollector
from torchrl.env import VecEnv


class OnPolicyCollectorBase(BaseCollector):
  def __init__(self, vf, discount=0.99, **kwargs):
    self.vf = vf
    super().__init__(**kwargs)
    self.discount = discount

  def take_actions(self):

    ob_tensor = torch.Tensor(
      self.current_ob
    ).to(self.device).unsqueeze(0)

    out = self.pf.explore(ob_tensor)
    act = out["action"]
    act = act.detach().cpu().numpy()

    value = self.vf(ob_tensor)
    value = value.cpu().item()

    if not self.continuous:
      act = act[0]
    elif np.isnan(act).any():
      print("NaN detected. BOOM")
      print(ob_tensor)
      print(self.pf.forward(ob_tensor))
      exit()

    next_ob, reward, done, info = self.env.step(act)
    if self.train_render:
      self.env.render()
    self.current_step += 1

    sample_dict = {
      "obs": np.expand_dims(self.current_ob, 0),
      "next_obs": np.expand_dims(next_ob, 0),
      "acts": np.expand_dims(act, 0),
      "values": [[value]],
      "rewards": [[reward]],
      "terminals": [[done]],
      "time_limits": [[
        info["time_limit"] if "time_limit" in info else False]]
    }
    self.train_rew += reward

    if done or self.current_step >= self.max_episode_frames:
      if not done and self.current_step >= self.max_episode_frames:
        last_ob = torch.Tensor(
          next_ob
        ).to(self.device).unsqueeze(0)
        last_value = self.vf(last_ob).detach().cpu().item()

        sample_dict["terminals"] = [True]
        sample_dict["rewards"] = [reward + self.discount * last_value]

      next_ob = self.env.reset()
      self.current_step = 0

      self.train_rews.append(self.train_rew)
      self.train_rew = 0
      # self.pf.finish_episode()
      # self.pf.start_episode()

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_ob

    return reward

  @property
  def funcs(self):
    return {
      "pf": self.pf,
      "vf": self.vf
    }


class VecOnPolicyCollector(VecCollector):
  def __init__(self, vf, discount=0.99, **kwargs):
    self.vf = vf
    super().__init__(**kwargs)
    self.discount = discount

  def take_actions(self):
    ob_tensor = torch.Tensor(
      self.current_ob
    ).to(self.device)

    out = self.pf.explore(ob_tensor)
    acts = out["action"]
    acts = acts.detach().cpu().numpy()

    values = self.vf(ob_tensor)
    values = values.detach().cpu().numpy()

    if type(acts) is not int:
      if np.isnan(acts).any():
        print("NaN detected. BOOM")
        print(ob_tensor)
        print(self.pf.forward(ob_tensor))
        exit()

    next_obs, rewards, dones, infos = self.env.step(acts)

    if self.train_render:
      self.env.render()
    self.current_step += 1

    sample_dict = {
      "obs": self.current_ob,
      "next_obs": next_obs,
      "acts": acts,
      "values": values,
      "rewards": rewards,
      "terminals": dones,
      "time_limits":
      infos["time_limit"][:, np.newaxis]
      if "time_limit" in infos else [False]
    }
    self.train_rew += rewards

    if np.any(dones):
      self.train_rews += list(self.train_rew[dones])
      self.train_rew[dones] = 0

    if np.any(dones) or \
       np.any(self.current_step >= self.max_episode_frames):

      surpass_flag = self.current_step >= self.max_episode_frames
      last_ob = torch.Tensor(
        next_obs
      ).to(self.device)

      last_value = self.vf(last_ob).detach().cpu().numpy()
      sample_dict["terminals"] = dones | surpass_flag
      sample_dict["rewards"] = rewards + \
        self.discount * last_value * surpass_flag

      next_obs = self.env.partial_reset(
        np.squeeze(dones | surpass_flag, axis=-1)
      )
      self.current_step[dones | surpass_flag] = 0
      self.train_rew[dones | surpass_flag] = 0

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_obs

    return np.sum(rewards)
