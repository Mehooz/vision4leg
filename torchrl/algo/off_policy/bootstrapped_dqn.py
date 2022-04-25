import numpy as np
import torch
from torch import nn as nn
from .dqn import DQN


class BootstrappedDQN(DQN):
  def __init__(self,
               head_num=10,
               bernoulli_p=0.5,
               **kwargs):
    super(BootstrappedDQN, self).__init__(**kwargs)

    # self.m_distribution = torch.distributions.bernoulli.Bernoulli( bernoulli_p )
    self.bernoulli_p = bernoulli_p
    self.head_num = head_num

    self.sample_key = ["obs", "next_obs", "actions",
                       "rewards", "terminals", "masks"]

  def take_actions(self, ob, action_func):

    action = action_func(ob)

    next_ob, reward, done, info = self.env.step(action)

    self.current_step += 1

    mask = np.random.binomial(
      n=1,
      p=self.bernoulli_p,
      size=self.head_num)

    sample_dict = {
      "obs": ob,
      "next_obs": next_ob,
      "acts": action,
      "rewards": [reward],
      "terminals": [done],
      "masks": mask
    }

    if done or self.current_step >= self.max_episode_frames:
      next_ob = self.env.reset()
      self.finish_episode()
      self.start_episode()
      self.current_step = 0

    self.replay_buffer.add_sample(sample_dict)

    return next_ob, done, reward, info

  def start_episode(self):
    self.pf.sample_head()

  def update(self, batch):
    self.training_update_num += 1

    obs = batch['obs']
    actions = batch['acts']
    next_obs = batch['next_obs']
    rewards = batch['rewards']
    terminals = batch['terminals']
    masks = batch['masks']

    obs = torch.Tensor(obs).to(self.device)
    actions = torch.Tensor(actions).to(self.device)
    next_obs = torch.Tensor(next_obs).to(self.device)
    rewards = torch.Tensor(rewards).to(self.device)
    terminals = torch.Tensor(terminals).to(self.device)
    masks = torch.Tensor(masks).to(self.device)

    mse_losses = []
    q_pred_all = self.qf(obs, range(self.head_num))
    next_q_pred_all = self.target_qf(next_obs, range(self.head_num))

    for i in range(self.head_num):
      q_pred = q_pred_all[i]
      q_s_a = q_pred.gather(1, actions.unsqueeze(1).long())
      next_q_pred = next_q_pred_all[i]

      target_q_s_a = rewards + self.discount * \
        (1 - terminals) * next_q_pred.max(1, keepdim=True)[0]
      # qf_loss = self.qf_criterion( q_s_a, target_q_s_a.detach())
      assert q_s_a.shape == target_q_s_a.shape
      mse_loss = (q_s_a - target_q_s_a) ** 2
      mse_losses.append(mse_loss)

    mse_losses = torch.cat(mse_losses, dim=1)
    qf_loss = (mse_losses * masks / self.head_num).sum(1).mean()

    self.qf_optimizer.zero_grad()
    qf_loss.backward()
    self.qf_optimizer.step()

    self._update_target_networks()

    # Information For Logger
    info = {}
    info['Reward_Mean'] = rewards.mean().item()
    info['Training/qf_loss'] = qf_loss.item()
    return info
