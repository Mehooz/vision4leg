import torch
import numpy as np
import copy
from .on_policy import VecOnPolicyCollector
from torchrl.env import VecEnv


pi = torch.acos(torch.zeros(1)).item() * 2


class VecOnPolicyHierarchicalCollector(VecOnPolicyCollector):
  def __init__(self, low_level_pf, **kwargs):
    self.low_level_pf = low_level_pf
    super().__init__(**kwargs)
    # self.state_dim = state_dim

  def take_actions(self):
    ob_tensor = torch.Tensor(
      self.current_ob
    ).to(self.device)

    out = self.pf.explore(ob_tensor, return_state=True)
    acts = out["action"]
    state = out["state"]
    # acts = acts.detach().cpu().numpy()

    # torch.
    angle = acts * pi * 0.5
    low_obs_dir = torch.cat([
      torch.cos(angle),
      torch.sin(angle)
    ], dim=-1)

    low_obs = torch.cat([
      low_obs_dir, state
    ], dim=-1)
    low_level_acts = self.low_level_pf.eval_act(low_obs)

    values = self.vf(ob_tensor)
    values = values.detach().cpu().numpy()

    acts = acts.detach().cpu().numpy()

    if type(acts) is not int:
      if np.isnan(acts).any():
        print("NaN detected. BOOM")
        print(ob_tensor)
        print(self.pf.forward(ob_tensor))
        exit()

    next_obs, rewards, dones, infos = self.env.step(low_level_acts)

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

    self.replay_buffer.add_sample(sample_dict)

    self.current_ob = next_obs

    return np.sum(rewards)

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

        act, state = self.pf.eval_act(
          torch.Tensor(eval_obs).to(self.device), return_state=True
        )
        act = torch.Tensor(act).to(self.device)
        # torch.
        angle = act * pi * 0.5
        low_obs_dir = torch.cat([
          torch.cos(angle),
          torch.sin(angle)
        ], dim=-1)

        low_obs = torch.cat([
          low_obs_dir, state
        ], dim=-1)
        low_level_acts = self.low_level_pf.eval_act(low_obs)

        if self.continuous and np.isnan(low_level_acts).any():
          print("NaN detected. BOOM")
          print(self.pf.forward(torch.Tensor(eval_obs).to(self.device)))
          exit()
        try:
          eval_obs, r, done, _ = self.eval_env.step(low_level_acts)
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

  @property
  def funcs(self):
    return {
      "pf": self.pf,
      "low_level_pf": self.low_level_pf
    }
