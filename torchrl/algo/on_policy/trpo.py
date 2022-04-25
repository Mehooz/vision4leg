import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.nn.utils.convert_parameters import parameters_to_vector
import numpy as np
from .a2c import A2C
import torchrl.algo.utils as atu


class TRPO(A2C):
  """
  TRPO
  """

  def __init__(
      self, max_kl, cg_damping, v_opt_times,
      cg_iters, residual_tol, **kwargs):
    super().__init__(**kwargs)

    self.max_kl = max_kl
    self.cg_damping = cg_damping
    self.cg_iters = cg_iters
    self.residual_tol = residual_tol
    self.v_opt_times = v_opt_times
    self.vf_sample_key = ["obs", "estimate_returns"]

  def mean_kl_divergence(self, model):
    """
    Returns an estimate of the average KL divergence
    between a given model and self.policy_model
    """
    def normal_distribution_kl_divergence(mean_old, std_old,
                                          mean_new, std_new):
      return torch.mean(torch.sum(
        (torch.log(std_new) - torch.log(std_old) +
         (std_old * std_old + (mean_old - mean_new).pow(2)) /
         (2.0 * std_new * std_new) - 0.5), 1))

    output_new = model.update(self.obs, self.acts)
    output_old = self.pf.update(self.obs, self.acts)

    if self.continuous:
      mean_new, std_new = output_new["mean"], output_new["std"]
      mean_old, std_old = output_old["mean"], output_old["std"]

      mean_new = mean_new.detach()
      std_new = std_new.detach()

      kl = normal_distribution_kl_divergence(
        mean_old, std_old, mean_new, std_new)
    else:
      probs_new = output_new["dis"]
      probs_old = output_old["dis"]

      probs_new = probs_new.detach()

      kl = torch.sum(
        probs_old * torch.log(
          probs_old / (probs_new + 1e-8)), 1).mean()

    return kl

  def hessian_vector_product(self, vector):
    """
    Returns the product of the Hessian of
    the KL divergence and the given vector
    """
    self.pf.zero_grad()
    mean_kl_div = self.mean_kl_divergence(self.pf)

    kl_grad_vector = torch.autograd.grad(
      mean_kl_div, self.pf.parameters(), create_graph=True)

    kl_grad_vector = torch.cat(
      [grad.view(-1) for grad in kl_grad_vector])
    grad_vector_product = torch.sum(kl_grad_vector * vector)

    second_order_grad = torch.autograd.grad(
      grad_vector_product, self.pf.parameters())

    fisher_vector_product = torch.cat(
      [grad.contiguous().view(-1) for grad in second_order_grad])

    return fisher_vector_product + self.cg_damping * vector.detach()

  def conjugate_gradient(self, b):
    """
    Returns F^(-1) b where F is the Hessian of the KL divergence
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(p).to(self.device)
    rdotr = r.double().dot(r.double())

    for _ in range(self.cg_iters):
      z = self.hessian_vector_product(p).squeeze(0)
      v = (rdotr / p.double().dot(z.double())).float()

      x += v * p
      r -= v * z

      newrdotr = r.double().dot(r.double())
      mu = newrdotr / rdotr

      p = r + mu.float() * p
      rdotr = newrdotr
      if rdotr < self.residual_tol:
        break
    return x

  def surrogate_loss(self, theta):
    """
    Returns the surrogate loss w.r.t. the given parameter vector theta
    """
    theta = theta.detach()
    new_model = copy.deepcopy(self.pf)
    vector_to_parameters(theta, new_model.parameters())

    output_new = new_model.update(self.obs, self.acts)
    output_old = self.pf.update(self.obs, self.acts)

    log_probs_new = output_new["log_prob"].detach()
    log_probs_old = output_old["log_prob"]

    ratio = torch.exp(log_probs_new - log_probs_old).detach()

    return -torch.mean(ratio * self.advs)

  def linesearch(self, x, fullstep, expected_improve_rate):
    """
    Returns the parameter vector given by a linesearch
    """
    accept_ratio = .1
    max_backtracks = 10
    fval = self.surrogate_loss(x)
    for n_backtrack, stepfrac in enumerate(.5**np.arange(max_backtracks)):
      print("Search number {}...".format(n_backtrack + 1))
      stepfrac = float(stepfrac)
      xnew = x + stepfrac * fullstep
      newfval = self.surrogate_loss(xnew)
      actual_improve = fval - newfval

      expected_improve = expected_improve_rate * stepfrac

      ratio = actual_improve / expected_improve

      if ratio > accept_ratio and actual_improve > 0:
        return xnew
    return x.detach()

  def update(self, batch):
    self.training_update_num += 1

    info = {}

    self.obs = batch['obs']
    self.acts = batch['acts']
    self.advs = batch['advs']

    self.obs = torch.Tensor(self.obs).to(self.device)
    self.acts = torch.Tensor(self.acts).to(self.device)
    self.advs = torch.Tensor(self.advs).to(self.device)

    info['advs/mean'] = self.advs.mean().item()
    info['advs/std'] = self.advs.std().item()
    info['advs/max'] = self.advs.max().item()
    info['advs/min'] = self.advs.min().item()
    # Calculate Advantage & Normalize it
    self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-4)

    out = self.pf.update(self.obs, self.acts)
    log_probs = out['log_prob']
    ent = out['ent']

    probs_new = torch.exp(log_probs)
    probs_old = probs_new.detach() + 1e-8

    ratio = probs_new / probs_old

    surrogate_loss = - torch.mean(ratio * self.advs) - \
      self.entropy_coeff * ent.mean()

    # Calculate the gradient of the surrogate loss
    self.pf.zero_grad()
    surrogate_loss.backward()
    policy_gradient = parameters_to_vector(
      [p.grad for p in self.pf.parameters()]).squeeze(0).detach()

    # ensure gradient is not zero
    if policy_gradient.nonzero().size()[0]:
      # Use Conjugate gradient to calculate step direction
      step_direction = self.conjugate_gradient(-policy_gradient)
      # line search for step
      shs = .5 * step_direction.dot(
        self.hessian_vector_product(step_direction))

      lm = torch.sqrt(shs / self.max_kl)
      fullstep = step_direction / lm

      gdotstepdir = -policy_gradient.dot(step_direction)
      theta = self.linesearch(parameters_to_vector(
        self.pf.parameters()).detach(),
        fullstep, gdotstepdir / lm)
      # Update parameters of policy model
      old_model = copy.deepcopy(self.pf)
      old_model.load_state_dict(self.pf.state_dict())

      if any(np.isnan(theta.cpu().detach().numpy())):
        print("NaN detected. Skipping update...")
      else:
        vector_to_parameters(theta, self.pf.parameters())

      kl_old_new = self.mean_kl_divergence(old_model)
      print('KL:{:10} , Entropy:{:10}'.format(
        kl_old_new.item(), ent.mean().item()))
    else:
      print("Policy gradient is 0. Skipping update...")
      print(policy_gradient.shape)

    self.pf.zero_grad()

    info['Training/policy_loss'] = surrogate_loss.item()

    info['logprob/mean'] = log_probs.mean().item()
    info['logprob/std'] = log_probs.std().item()
    info['logprob/max'] = log_probs.max().item()
    info['logprob/min'] = log_probs.min().item()

    return info

  def update_vf(self, batch):
    self.training_update_num += 1

    obs = batch['obs']
    est_rets = batch['estimate_returns']

    obs = torch.Tensor(obs).to(self.device)
    est_rets = torch.Tensor(est_rets).to(self.device)

    values = self.vf(obs)
    assert values.shape == est_rets.shape, \
      print(values.shape, est_rets.shape)
    vf_loss = 0.5 * (values - est_rets).pow(2).mean()

    self.vf_optimizer.zero_grad()
    vf_loss.backward()
    vf_grad_norm = torch.nn.utils.clip_grad_norm_(
      self.vf.parameters(), 0.5)
    self.vf_optimizer.step()

    info = {}
    info['Training/vf_loss'] = vf_loss.item()
    info['grad_norm/vf'] = vf_grad_norm.item()

    return info

  def update_per_epoch(self):
    self.process_epoch_samples()
    atu.update_linear_schedule(
      self.pf_optimizer, self.current_epoch, self.num_epochs, self.plr)
    atu.update_linear_schedule(
      self.vf_optimizer, self.current_epoch, self.num_epochs, self.vlr)
    whole_batch = {
      "obs": self.replay_buffer._obs.copy(),
      "acts": self.replay_buffer._acts.copy(),
      "advs": self.replay_buffer._advs.copy(),
      "estimate_returns": self.replay_buffer._estimate_returns.copy()
    }
    infos = self.update(whole_batch)
    self.logger.add_update_info(infos)

    for _ in range(self.v_opt_times):
      for batch in self.replay_buffer.one_iteration(self.batch_size,
                                                    self.vf_sample_key,
                                                    self.shuffle):
        infos = self.update_vf(batch)
        self.logger.add_update_info(infos)

  @property
  def networks(self):
    return [
      self.pf,
      self.vf
    ]
