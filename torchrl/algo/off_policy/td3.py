import numpy as np
import copy
import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import Normal
from .off_rl_algo import OffRLAlgo


class TD3(OffRLAlgo):
  def __init__(
    self,
    pf, qf1, qf2,
    plr, qlr,
    optimizer_class=optim.Adam,

      policy_update_delay=2,
      norm_std_policy=0.2,
      noise_clip=0.5,
      **kwargs):
    super(TD3, self).__init__(**kwargs)

    self.pf = pf
    self.target_pf = copy.deepcopy(pf)

    self.qf1 = qf1
    self.target_qf1 = copy.deepcopy(qf1)
    self.qf2 = qf2
    self.target_qf2 = copy.deepcopy(qf2)
    self.to(self.device)

    self.plr = plr
    self.qlr = qlr

    self.pf_optimizer = optimizer_class(
      self.pf.parameters(),
      lr=self.plr,
    )

    self.qf1_optimizer = optimizer_class(
      self.qf1.parameters(),
      lr=self.qlr,
    )

    self.qf2_optimizer = optimizer_class(
      self.qf2.parameters(),
      lr=self.qlr,
    )

    self.qf_criterion = nn.MSELoss()

    self.policy_update_delay = policy_update_delay

    self.norm_std_policy = norm_std_policy
    self.noise_clip = noise_clip

  def update(self, batch):
    self.training_update_num += 1

    obs = batch['obs']
    actions = batch['acts']
    next_obs = batch['next_obs']
    rewards = batch['rewards']
    terminals = batch['terminals']

    obs = torch.Tensor(obs).to(self.device)
    actions = torch.Tensor(actions).to(self.device)
    next_obs = torch.Tensor(next_obs).to(self.device)
    rewards = torch.Tensor(rewards).to(self.device)
    terminals = torch.Tensor(terminals).to(self.device)

    """
        QF Loss
        """
    sample_info = self.target_pf.explore(next_obs)
    target_actions = sample_info["action"]

    noise = Normal(
      torch.zeros(target_actions.size()),
      self.norm_std_policy * torch.ones(target_actions.size())
    ).sample().to(target_actions.device)
    noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
    target_actions += noise
    target_actions = torch.clamp(target_actions, -1, 1)

    target_q_values = torch.min(
      self.target_qf1([next_obs, target_actions]),
      self.target_qf2([next_obs, target_actions]))

    q_target = rewards + (1. - terminals) * self.discount * target_q_values
    q1_pred = self.qf1([obs, actions])
    q2_pred = self.qf2([obs, actions])

    assert q1_pred.shape == q_target.shape
    assert q2_pred.shape == q_target.shape
    qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
    qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

    self.qf1_optimizer.zero_grad()
    qf1_loss.backward()
    if self.grad_clip:
      qf1_grad_norm = torch.nn.utils.clip_grad_norm_(
        self.qf1.parameters(), self.grad_clip)
    self.qf1_optimizer.step()

    self.qf2_optimizer.zero_grad()
    qf2_loss.backward()
    if self.grad_clip:
      qf2_grad_norm = torch.nn.utils.clip_grad_norm_(
        self.qf2.parameters(), self.grad_clip)
    self.qf2_optimizer.step()

    # Information For Logger
    info = {}

    info['Reward_Mean'] = rewards.mean().item()

    info['Training/qf1_loss'] = qf1_loss.item()
    info['Training/qf2_loss'] = qf2_loss.item()
    if self.grad_clip is not None:
      info['Training/qf1_grad_norm'] = qf1_grad_norm.item()
      info['Training/qf2_grad_norm'] = qf2_grad_norm.item()

    if self.training_update_num % self.policy_update_delay:
      """
      Policy Loss.
      """
      new_actions = self.pf(obs)
      new_q_pred_1 = self.qf1([obs, new_actions])
      policy_loss = -new_q_pred_1.mean()

      """
            Update Networks
            """

      self.pf_optimizer.zero_grad()
      policy_loss.backward()
      if self.grad_clip:
        pf_grad_norm = torch.nn.utils.clip_grad_norm_(
          self.pf.parameters(), self.grad_clip)
      self.pf_optimizer.step()

      self._update_target_networks()

      info['Training/policy_loss'] = policy_loss.item()
      if self.grad_clip is not None:
        info['Training/pf_grad_norm'] = pf_grad_norm.item()

      info['new_actions/mean'] = new_actions.mean().item()
      info['new_actions/std'] = new_actions.std().item()
      info['new_actions/max'] = new_actions.max().item()
      info['new_actions/min'] = new_actions.min().item()

    return info

  @property
  def networks(self):
    return [
      self.pf,
      self.qf1,
      self.qf2,
      self.target_pf,
      self.target_qf1,
      self.target_qf2
    ]

  @property
  def snapshot_networks(self):
    return [
      ["pf", self.pf],
      ["qf1", self.qf1],
      ["qf2", self.qf2]
    ]

  @property
  def target_networks(self):
    return [
      (self.pf, self.target_pf),
      (self.qf1, self.target_qf1),
      (self.qf2, self.target_qf2),
    ]
