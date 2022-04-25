import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torchrl.networks as networks


class UniformPolicyDiscrete(nn.Module):
  def __init__(self, action_num):
    super().__init__()
    self.action_num = action_num
    self.continuous = False

  def forward(self, x):
    return np.random.randint(self.action_num)

  def explore(self, x):
    return {
      "action": np.random.randint(self.action_num)
    }


class EpsilonGreedyDQNDiscretePolicy():
  """
  wrapper over QNet
  """

  def __init__(
      self, qf,
      start_epsilon, end_epsilon,
      decay_frames, action_shape):
    self.qf = qf
    self.start_epsilon = start_epsilon
    self.end_epsilon = end_epsilon
    self.decay_frames = decay_frames
    self.count = 0
    self.action_shape = action_shape
    self.epsilon = self.start_epsilon
    self.continuous = False

  def q_to_a(self, q):
    return q.max(dim=-1, keepdim=True)[1].detach()

  def explore(self, x):
    self.count += 1
    x = x.squeeze(0)

    if self.count < self.decay_frames:
      self.epsilon = self.start_epsilon - \
        (self.start_epsilon - self.end_epsilon) \
        * (self.count / self.decay_frames)
    else:
      self.epsilon = self.end_epsilon

    output = self.qf(x)
    action = self.q_to_a(output)
    r = torch.Tensor(np.random.rand(*action.shape))
    random_action = torch.LongTensor(
      np.random.randint(
        low=0, high=self.action_shape, size=action.shape)
    ).to(x.device)

    action[r < self.epsilon] = random_action[r < self.epsilon]

    return {
      "q_value": output,
      "action": action
    }

  def eval_act(self, x):
    output = self.qf(x)
    action = self.q_to_a(output)
    return action.cpu().numpy()

  def to(self, device):
    self.qf.to(device)


class EpsilonGreedyQRDQNDiscretePolicy(EpsilonGreedyDQNDiscretePolicy):
  """
  wrapper over DRQNet
  """

  def __init__(self, quantile_num, **kwargs):
    super().__init__(**kwargs)
    self.quantile_num = quantile_num
    self.continuous = False

  def q_to_a(self, q):
    q = q.view(-1, self.action_shape, self.quantile_num)
    return q.mean(dim=-1).max(dim=-1)[1].detach().item()


class BootstrappedDQNDiscretePolicy():
  """
  wrapper over Bootstrapped QNet
  """

  def __init__(self, qf, head_num, action_shape):
    self.qf = qf
    self.head_num = head_num
    self.action_shape = action_shape
    self.idx = 0
    self.continuous = False

  def sample_head(self):
    self.idx = np.random.randint(self.head_num)

  def set_head(self, idx):
    self.idx = idx

  def explore(self, x):
    output = self.qf(x, [self.idx])
    action = output[0].max(dim=-1)[1].detach().item()
    return {
      "q_value": output[0],
      "action": action
    }

  def eval_act(self, x):
    output = self.qf(x, range(self.head_num))
    output = torch.mean(torch.cat(output, dim=0), dim=0)
    action = output.max(dim=-1)[1].detach().item()
    return action


class CategoricalDisPolicy(networks.Net):
  """
  Discrete Policy
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.continuous = False

  def forward(self, x):
    logits = super().forward(x)
    return torch.softmax(logits, dim=-1)

  def explore(self, x, return_log_probs=False):

    output = self.forward(x)
    dis = Categorical(output)
    action = dis.sample()

    out = {
      "dis": output,
      "action": action
    }

    if return_log_probs:
      out['log_prob'] = dis.log_prob(action)

    return out

  def eval_act(self, x):
    output = self.forward(x)
    return output.max(dim=-1)[1].detach().cpu().numpy()

  def update(self, obs, actions):
    output = self.forward(obs)
    dis = Categorical(output)

    log_prob = dis.log_prob(actions).unsqueeze(-1)
    ent = dis.entropy()

    out = {
      "dis": dis,
      "log_prob": log_prob,
      "ent": ent
    }
    return out
