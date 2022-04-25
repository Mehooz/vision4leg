import torch
import numpy as np

from .base import ParallelCollector


class ParallelOnPolicyCollector(ParallelCollector):
  def __init__(self, vf, discount=0.99, **kwargs):
    self.vf = vf
    self.vf.share_memory()
    self.discount = discount
    super().__init__(**kwargs)

  @classmethod
  def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

    pf = funcs["pf"]
    vf = funcs["vf"]

    ob = ob_info["ob"]
    ob_tensor = torch.Tensor(ob).to(env_info.device).unsqueeze(0)

    out = pf.explore(ob_tensor)
    act = out["action"]
    act = act.detach().cpu().numpy()

    value = vf(ob_tensor)
    value = value.item()

    if not env_info.continuous:
      act = act[0]

    if type(act) is not int:
      if np.isnan(act).any():
        print("NaN detected. BOOM")
        exit()

    next_ob, reward, done, info = env_info.env.step(act)
    if env_info.train_render:
      env_info.env.render()
    env_info.current_step += 1

    sample_dict = {
      "obs": ob,
      "next_obs": next_ob,
      "acts": act,
      "values": [value],
      "rewards": [reward],
      "terminals": [done]
    }

    if done or env_info.current_step >= env_info.max_episode_frames:
      if not done and env_info.current_step >= env_info.max_episode_frames:
        last_ob = torch.Tensor(next_ob).to(env_info.device).unsqueeze(0)
        last_value = env_info.vf(last_ob).item()

        sample_dict["terminals"] = [True]
        sample_dict["rewards"] = [reward + env_info.discount * last_value]

      next_ob = env_info.env.reset()
      env_info.finish_episode()
      env_info.start_episode()

    replay_buffer.add_sample(sample_dict, env_info.env_rank)

    return next_ob, done, reward, info

  @property
  def funcs(self):
    return {
      "pf": self.pf,
      "vf": self.vf
    }
