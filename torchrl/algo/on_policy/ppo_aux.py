import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from .a2c import A2C
import torchrl.algo.utils as atu


class PPOAux(A2C):
  """
  Actor Critic
  """

  def __init__(
      self, pf,
      clip_para=0.2,
      opt_epochs=10,
      clipped_value_loss=False,
      aux_coeff=1.,
      **kwargs):
    self.target_pf = copy.deepcopy(pf)
    super(PPOAux, self).__init__(pf=pf, **kwargs)
    self.aux_coeff = aux_coeff
    self.clip_para = clip_para
    self.opt_epochs = opt_epochs
    self.clipped_value_loss = clipped_value_loss
    self.sample_key = ["obs", "acts", "advs", "estimate_returns", "values"]

  def update_per_epoch(self):
    self.process_epoch_samples()
    atu.update_linear_schedule(
      self.pf_optimizer, self.current_epoch, self.num_epochs, self.plr)
    atu.update_linear_schedule(
      self.vf_optimizer, self.current_epoch, self.num_epochs, self.vlr)
    atu.copy_model_params_from_to(self.pf, self.target_pf)
    for _ in range(self.opt_epochs):
      for batch in self.replay_buffer.one_iteration(self.batch_size,
                                                    self.sample_key,
                                                    self.shuffle):
        infos = self.update(batch)
        self.logger.add_update_info(infos)

  def update_actor(
      self,
      info,
      obs,
      actions,
      advs
  ):

    out = self.pf.update(obs, actions)
    log_probs = out['log_prob']
    ent = out['ent']
    log_std = out['log_std']
    aux_loss = out['aux_loss']

    with torch.no_grad():
      target_out = self.target_pf.update(obs, actions)
      target_log_probs = target_out['log_prob']

    ratio = torch.exp(log_probs - target_log_probs.detach())

    assert ratio.shape == advs.shape, print(ratio.shape, advs.shape)
    surrogate_loss_pre_clip = ratio * advs
    surrogate_loss_clip = torch.clamp(ratio,
                                      1.0 - self.clip_para,
                                      1.0 + self.clip_para) * advs

    policy_loss = -torch.mean(torch.min(
      surrogate_loss_clip, surrogate_loss_pre_clip))
    policy_loss = policy_loss - self.entropy_coeff * \
      ent.mean() + self.aux_coeff * aux_loss

    self.pf_optimizer.zero_grad()
    policy_loss.backward()
    pf_grad_norm = torch.nn.utils.clip_grad_norm_(
      self.pf.parameters(), 0.5)
    self.pf_optimizer.step()

    info['Training/policy_loss'] = policy_loss.item()

    info['logprob/mean'] = log_probs.mean().item()
    info['logprob/std'] = log_probs.std().item()
    info['logprob/max'] = log_probs.max().item()
    info['logprob/min'] = log_probs.min().item()

    info['log_std/mean'] = log_std.mean().item()
    info['log_std/std'] = log_std.std().item()
    info['log_std/max'] = log_std.max().item()
    info['auxiliary/loss'] = aux_loss.item()

    info['log_std/min'] = log_std.min().item()

    info['ratio/max'] = ratio.max().item()
    info['ratio/min'] = ratio.min().item()

    info['grad_norm/pf'] = pf_grad_norm.item()

  def update_critic(
      self,
      info,
      obs,
      old_values,
      est_rets
  ):
    values = self.vf(obs)
    assert values.shape == est_rets.shape, \
      print(values.shape, est_rets.shape)

    if self.clipped_value_loss:
      values_clipped = old_values + \
        (values - old_values).clamp(-self.clip_para, self.clip_para)
      vf_loss = (values - est_rets).pow(2)
      vf_loss_clipped = (
        values_clipped - est_rets).pow(2)
      vf_loss = 0.5 * torch.max(vf_loss,
                                vf_loss_clipped).mean()
    else:
      vf_loss = self.vf_criterion(values, est_rets)

    self.vf_optimizer.zero_grad()
    vf_loss.backward()
    vf_grad_norm = torch.nn.utils.clip_grad_norm_(
      self.vf.parameters(), 0.5)
    self.vf_optimizer.step()

    info['Training/vf_loss'] = vf_loss.item()
    info['grad_norm/vf'] = vf_grad_norm.item()

  def update(self, batch):
    self.training_update_num += 1

    info = {}

    obs = batch['obs']
    actions = batch['acts']
    advs = batch['advs']
    old_values = batch['values']
    est_rets = batch['estimate_returns']

    obs = torch.Tensor(obs).to(self.device)
    actions = torch.Tensor(actions).to(self.device)
    advs = torch.Tensor(advs).to(self.device)
    old_values = torch.Tensor(old_values).to(self.device)
    est_rets = torch.Tensor(est_rets).to(self.device)

    info['advs/mean'] = advs.mean().item()
    info['advs/std'] = advs.std().item()
    info['advs/max'] = advs.max().item()
    info['advs/min'] = advs.min().item()

    # Normalize the advantage
    advs = (advs - advs.mean()) / (advs.std() + 1e-5)

    self.update_critic(info, obs, old_values, est_rets)
    self.update_actor(info, obs, actions, advs)

    return info

  @property
  def networks(self):
    return [
      self.pf,
      self.vf,
      self.target_pf
    ]
