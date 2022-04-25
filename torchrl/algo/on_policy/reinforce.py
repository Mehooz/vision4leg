import torch
import torch.optim as optim
from .on_rl_algo import OnRLAlgo
from torchrl.networks.nets import ZeroNet


class Reinforce(OnRLAlgo):
  """
  Reinforce
  """

  def __init__(
      self,
      pf,
      plr,
      optimizer_class=optim.Adam,
      entropy_coeff=0.001,
      **kwargs):
    super(Reinforce, self).__init__(**kwargs)
    self.pf = pf
    # Use a vacant value network to simplify the code structure
    self.vf = ZeroNet()
    self.to(self.device)

    self.plr = plr
    self.pf_optimizer = optimizer_class(
      self.pf.parameters(),
      lr=self.plr
    )

    self.entropy_coeff = entropy_coeff
    self.gae = False

  def update(self, batch):
    self.training_update_num += 1

    info = {}

    obs = batch['obs']
    acts = batch['acts']
    advs = batch['advs']

    info['advs/mean'] = advs.mean().item()
    info['advs/std'] = advs.std().item()
    info['advs/max'] = advs.max().item()
    info['advs/min'] = advs.min().item()

    obs = torch.Tensor(obs).to(self.device)
    acts = torch.Tensor(acts).to(self.device)
    advs = torch.Tensor(advs).to(self.device)

    out = self.pf.update(obs, acts)
    log_probs = out['log_prob']
    ent = out['ent']

    # Normalize the advantage
    advs = (advs - advs.mean()) / (advs.std() + 1e-5)

    assert log_probs.shape == advs.shape
    policy_loss = -log_probs * advs
    policy_loss = policy_loss.mean() - self.entropy_coeff * ent.mean()

    self.pf_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.pf.parameters(), 0.5)
    self.pf_optimizer.step()

    info['Training/policy_loss'] = policy_loss.item()

    info['ent'] = ent.mean().item()
    info['logprob/mean'] = log_probs.mean().item()
    info['logprob/std'] = log_probs.std().item()
    info['logprob/max'] = log_probs.max().item()
    info['logprob/min'] = log_probs.min().item()

    return info

  @property
  def snapshot_networks(self):
    return [
      ("pf", self.pf)
    ]
