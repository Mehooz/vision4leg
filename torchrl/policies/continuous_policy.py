import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torchrl.networks as networks
from .distribution import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5


class UniformPolicyContinuous(nn.Module):
  def __init__(self, action_shape):
    super().__init__()
    self.continuous = True
    self.action_shape = action_shape

  def forward(self, x):
    return torch.Tensor(np.random.uniform(-1., 1., self.action_shape))

  def explore(self, x):
    return {
      "action": torch.Tensor(np.random.uniform(
        -1., 1., self.action_shape))
    }


class DetContPolicy(networks.Net):
  def __init__(self, tanh_action=False, **kwargs):
    super().__init__(**kwargs)
    self.continuous = True
    self.tanh_action = tanh_action

  def forward(self, x):
    if self.tanh_action:
      return torch.tanh(super().forward(x))
    else:
      return super().forward(x)

  def eval_act(self, x):
    with torch.no_grad():
      return self.forward(x).squeeze(0).detach().cpu().numpy()

  def explore(self, x):
    return {
      "action": self.forward(x).squeeze(0)
    }


class FixGaussianContPolicy(networks.Net):
  def __init__(self, norm_std_explore, tanh_action=False, **kwargs):
    super().__init__(**kwargs)
    self.continuous = True
    self.tanh_action = tanh_action
    self.norm_std_explore = norm_std_explore

  def forward(self, x):
    if self.tanh_action:
      return torch.tanh(super().forward(x))
    else:
      return super().forward(x)

  def eval_act(self, x):
    with torch.no_grad():
      return self.forward(x).squeeze(0).detach().cpu().numpy()

  def explore(self, x):
    action = self.forward(x).squeeze(0)
    noise = Normal(
      0, self.norm_std_explore).sample(action.shape).to(action.device)
    action = action + noise
    return {
      "action": action
    }


class GaussianContPolicyBase():
  def eval_act(self, x):
    with torch.no_grad():
      mean, _, _ = self.forward(x)
    if self.tanh_action:
      mean = torch.tanh(mean)
    return mean.squeeze(0).detach().cpu().numpy()

  def explore(self, x, return_log_probs=False, return_pre_tanh=False):
    mean, std, log_std = self.forward(x)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    ent = dis.entropy().sum(-1, keepdim=True)

    dic = {
      "mean": mean,
      "log_std": log_std,
      "std": std,
      "ent": ent
    }

    if return_log_probs:
      if self.tanh_action:
        action, z = dis.rsample(return_pretanh_value=True)
        log_prob = dis.log_prob(
          action,
          pre_tanh_value=z
        )
        dic["pre_tanh"] = z.squeeze(0)
      else:
        action = dis.sample()
        log_prob = dis.log_prob(action)
      log_prob = log_prob.sum(dim=-1, keepdim=True)
      dic["log_prob"] = log_prob
    else:
      if self.tanh_action:
        if return_pre_tanh:
          action, z = dis.rsample(return_pretanh_value=True)
          dic["pre_tanh"] = z.squeeze(0)
        action = dis.rsample(return_pretanh_value=False)
      else:
        action = dis.sample()

    dic["action"] = action.squeeze(0)
    return dic

  def update(self, obs, actions):
    mean, std, log_std = self.forward(obs)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
    ent = dis.entropy().sum(-1, keepdim=True)

    out = {
      "mean": mean,
      "dis": Normal(mean, std),
      "log_std": log_std,
      "std": std,
      "log_prob": log_prob,
      "ent": ent
    }
    return out


class GaussianContPolicyWithAux():
  def eval_act(self, x):
    with torch.no_grad():
      mean, _, _ = self.forward(x)
    if self.tanh_action:
      mean = torch.tanh(mean)
    return mean.squeeze(0).detach().cpu().numpy()

  def explore(self, x, return_log_probs=False, return_pre_tanh=False):
    mean, std, log_std = self.forward(x)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    ent = dis.entropy().sum(-1, keepdim=True)

    dic = {
      "mean": mean,
      "log_std": log_std,
      "std": std,
      "ent": ent
    }

    if return_log_probs:
      if self.tanh_action:
        action, z = dis.rsample(return_pretanh_value=True)
        log_prob = dis.log_prob(
          action,
          pre_tanh_value=z
        )
        dic["pre_tanh"] = z.squeeze(0)
      else:
        action = dis.sample()
        log_prob = dis.log_prob(action)
      log_prob = log_prob.sum(dim=-1, keepdim=True)
      dic["log_prob"] = log_prob
    else:
      if self.tanh_action:
        if return_pre_tanh:
          action, z = dis.rsample(return_pretanh_value=True)
          dic["pre_tanh"] = z.squeeze(0)
        action = dis.rsample(return_pretanh_value=False)
      else:
        action = dis.sample()

    dic["action"] = action.squeeze(0)
    return dic

  def update(self, obs, actions):
    mean, std, log_std, loss = self.forward_and_compute_aux_loss(obs)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
    ent = dis.entropy().sum(-1, keepdim=True)

    out = {
      "mean": mean,
      "dis": Normal(mean, std),
      "log_std": log_std,
      "std": std,
      "log_prob": log_prob,
      "ent": ent,
      "aux_loss": loss
    }
    return out


class GaussianContPolicy(networks.Net, GaussianContPolicyBase):
  def __init__(self, tanh_action=False, **kwargs):
    super().__init__(**kwargs)
    self.continuous = True
    self.tanh_action = tanh_action

  def forward(self, x):
    x = super().forward(x)

    mean, log_std = x.chunk(2, dim=-1)

    log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(log_std)

    return mean, std, log_std


class GaussianContPolicyBasicBias(networks.Net, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd


class GaussianContPolicyNatureEncoderProj(networks.NatureEncoderProjNet, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd


class GaussianContPolicyImpalaEncoderProj(networks.ImpalaEncoderProjNet, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd


class GaussianContPolicyImpalaFuseResidual(networks.ImpalaFuseResidualActor, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd


class GaussianContPolicyImpalaFuseResidualWithAux(networks.ImpalaFuseResidualActor, GaussianContPolicyWithAux):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd

  def forward_and_compute_aux_loss(self, x):
    mean, aux_loss = super().forward_and_compute_aux_loss(x)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd, aux_loss


class GaussianContPolicyImpalaWeightedFuseResidual(networks.ImpalaWeightedFuseResidualActor, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd


class GaussianContPolicyVis(networks.VisualNet, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x, return_state=False):
    if return_state:
      mean, state = super().forward(x, return_state=return_state)
    else:
      mean = super().forward(x, return_state=return_state)

    # logstd = torch.clamp(self.logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    if return_state:
      return mean, std, logstd, state
    return mean, std, logstd

  def eval_act(self, x, return_state=False):
    with torch.no_grad():
      if return_state:
        mean, _, _, state = self.forward(x, return_state=return_state)
      else:
        mean, _, _ = self.forward(x, return_state=return_state)
    if self.tanh_action:
      mean = torch.tanh(mean)
    if return_state:
      return mean.squeeze(0).detach().cpu().numpy(), state
    return mean.squeeze(0).detach().cpu().numpy()

  def explore(self, x, return_log_probs=False, return_pre_tanh=False, return_state=False):
    if return_state:
      mean, std, log_std, state = self.forward(
        x, return_state=return_state)
    else:
      mean, std, log_std = self.forward(x)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    ent = dis.entropy().sum(-1, keepdim=True)

    dic = {
      "mean": mean,
      "log_std": log_std,
      "std": std,
      "ent": ent
    }
    if return_state:
      dic["state"] = state

    if return_log_probs:
      if self.tanh_action:
        action, z = dis.rsample(return_pretanh_value=True)
        log_prob = dis.log_prob(
          action,
          pre_tanh_value=z
        )
        dic["pre_tanh"] = z.squeeze(0)
      else:
        action = dis.sample()
        log_prob = dis.log_prob(action)
      log_prob = log_prob.sum(dim=-1, keepdim=True)
      dic["log_prob"] = log_prob
    else:
      if self.tanh_action:
        if return_pre_tanh:
          action, z = dis.rsample(return_pretanh_value=True)
          dic["pre_tanh"] = z.squeeze(0)
        action = dis.rsample(return_pretanh_value=False)
      else:
        action = dis.sample()

    dic["action"] = action.squeeze(0)
    return dic

  def update(self, obs, actions):
    mean, std, log_std = self.forward(obs, return_state=False)

    if self.tanh_action:
      dis = TanhNormal(mean, std)
    else:
      dis = Normal(mean, std)

    log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
    ent = dis.entropy().sum(-1, keepdim=True)

    out = {
      "mean": mean,
      "dis": Normal(mean, std),
      "log_std": log_std,
      "std": std,
      "log_prob": log_prob,
      "ent": ent
    }
    return out


class GaussianContPolicyTransformer(
    networks.Transformer, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd


class GaussianContPolicyLocoTransformer(
    networks.LocoTransformer, GaussianContPolicyBase):
  def __init__(self, output_shape, tanh_action=False, log_init=0.125, **kwargs):
    super().__init__(output_shape=output_shape, **kwargs)
    self.continuous = True
    self.logstd = nn.Parameter(torch.ones(output_shape) * np.log(log_init))
    self.tanh_action = tanh_action

  def forward(self, x):
    mean = super().forward(x)
    logstd = self.logstd
    logstd = torch.clamp(logstd, LOG_SIG_MIN, LOG_SIG_MAX)
    std = torch.exp(logstd)
    std = std.unsqueeze(0).expand_as(mean)
    return mean, std, logstd
