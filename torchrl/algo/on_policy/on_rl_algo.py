import numpy as np
import torch
from torchrl.algo.rl_algo import RLAlgo


class OnRLAlgo(RLAlgo):
  """
  Base RL Algorithm Framework
  """

  def __init__(
      self,
      shuffle=True,
      tau=None,
      gae=True,
      **kwargs):
    super(OnRLAlgo, self).__init__(**kwargs)
    self.sample_key = ["obs", "acts", "advs", "estimate_returns"]
    self.shuffle = shuffle
    self.tau = tau
    self.gae = gae

  def process_epoch_samples(self):
    sample = self.replay_buffer.last_sample(
      ['next_obs', 'terminals', "time_limits"])
    last_ob = torch.Tensor(sample['next_obs']).to(self.device)
    last_value = self.vf(last_ob).detach().cpu().numpy()
    last_value = last_value * (1 - sample["terminals"])
    if self.gae:
      self.replay_buffer.generalized_advantage_estimation(last_value,
                                                          self.discount,
                                                          self.tau)
    else:
      self.replay_buffer.discount_reward(last_value, self.discount)

  def update_per_epoch(self):
    self.process_epoch_samples()
    for batch in self.replay_buffer.one_iteration(
        self.batch_size, self.sample_key, self.shuffle):
      infos = self.update(batch)
      self.logger.add_update_info(infos)

  @property
  def networks(self):
    return [
      self.pf,
      self.vf
    ]
