import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .a2c import A2C
import torchrl.algo.utils as atu


class VMPO(A2C):
  """
  Actor Critic
  """

  def __init__(
    self, pf,
    # clip_para=0.2,
    opt_epochs=10,
    eta_eps=0.02,
    alpha_eps=0.1,
    clipped_value_loss=False,
      **kwargs):
    self.target_pf = copy.deepcopy(pf)
    super(VMPO, self).__init__(pf=pf, **kwargs)

    self.eta_eps = eta_eps
    self.eta = torch.Tensor([1]).to(self.device)
    self.eta.requires_grad_()

    self.alpha_eps = alpha_eps
    self.alpha = torch.Tensor([0.1]).to(self.device)
    self.alpha.requires_grad_()

    self.param_optimizer = self.optimizer_class(
      [self.eta, self.alpha],
      lr=self.plr,
      eps=1e-5,
    )

    self.opt_epochs = opt_epochs
    self.sample_key = ["obs", "acts", "advs", "estimate_returns", "values"]

  def update_per_epoch(self):
    self.process_epoch_samples()
    # atu.update_linear_schedule(
    #     self.pf_optimizer, self.current_epoch, self.num_epochs, self.plr)
    # atu.update_linear_schedule(
    #     self.vf_optimizer, self.current_epoch, self.num_epochs, self.vlr)
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
      advs,
  ):
    _, idx = torch.sort(advs, dim=0, descending=True)
    idx = idx.reshape(-1).long()
    idx, _ = idx.chunk(2, dim=0)

    obs = obs[idx, ...]
    actions = actions[idx, ...]
    advs = advs[idx, ...]
    # print(obs, actions, advs)
    # print(advs)

    out = self.pf.update(obs, actions)
    log_probs = out['log_prob']
    dis = out["dis"]

    log_probs = log_probs

    with torch.no_grad():
      target_out = self.target_pf.update(obs, actions)
      target_log_probs = target_out['log_prob']
      target_log_probs = target_log_probs
      target_dis = target_out["dis"]

    phis = F.softmax(advs/self.eta.detach(), dim=0)

    policy_loss = -phis * log_probs
    eta_loss = self.eta * self.eta_eps + \
      self.eta * torch.log(
        torch.mean(torch.exp(advs/self.eta))
      )

    kl = torch.distributions.kl.kl_divergence(
      dis, target_dis).sum(-1, keepdim=True)

    alpha_loss = self.alpha * self.alpha_eps - self.alpha * kl.detach().mean()

    policy_loss += self.alpha.detach() * kl
    policy_loss = policy_loss.mean()
    loss = policy_loss + eta_loss + alpha_loss

    self.pf_optimizer.zero_grad()
    self.param_optimizer.zero_grad()

    loss.backward()

    pf_grad_norm = torch.nn.utils.clip_grad_norm_(
      self.pf.parameters(), 0.5)

    self.pf_optimizer.step()
    self.param_optimizer.step()

    with torch.no_grad():
      self.eta.copy_(torch.clamp(self.eta, min=1e-8))
      self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))

    info['Training/policy_loss'] = policy_loss.item()
    info['Training/alpha_loss'] = alpha_loss.item()
    info['Training/alpha'] = self.alpha.item()
    info['Training/eta'] = self.eta.item()

    info['logprob/mean'] = log_probs.mean().item()
    info['logprob/std'] = log_probs.std().item()
    info['logprob/max'] = log_probs.max().item()
    info['logprob/min'] = log_probs.min().item()

    info['KL/mean'] = kl.detach().mean().item()
    info['KL/std'] = kl.detach().std().item()
    info['KL/max'] = kl.detach().max().item()
    info['KL/min'] = kl.detach().min().item()

    info['grad_norm/pf'] = pf_grad_norm.item()

  def update_critic(
      self,
      info,
      obs,
      est_rets
  ):

    values = self.vf(obs)
    assert values.shape == est_rets.shape, \
      print(values.shape, est_rets.shape)
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

    self.update_critic(info, obs, est_rets)
    self.update_actor(info, obs, actions, advs)

    return info

  @property
  def networks(self):
    return [
      self.pf,
      self.vf,
      self.target_pf
    ]
