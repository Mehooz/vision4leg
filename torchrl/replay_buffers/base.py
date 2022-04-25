import numpy as np


class BaseReplayBuffer():
  """
  Basic Replay Buffer
  """

  def __init__(
      self,
      max_replay_buffer_size,
      env_nums=1,
      time_limit_filter=False):
    self.env_nums = env_nums
    self._max_replay_buffer_size = max_replay_buffer_size // self.env_nums
    self._top = 0
    self._size = 0
    self.time_limit_filter = time_limit_filter

  def add_sample(self, sample_dict, **kwargs):
    for key in sample_dict:
      if not hasattr(self, "_" + key):
        # do not add env_nums dimension here,
        # since it's included in data itself
        self.__setattr__(
          "_" + key,
          np.zeros((self._max_replay_buffer_size,) +
                   np.shape(sample_dict[key])))
      self.__getattribute__("_" + key)[self._top, ...] = sample_dict[key]
    self._advance()

  def terminate_episode(self):
    pass

  def _advance(self):
    self._top = (self._top + 1) % self._max_replay_buffer_size
    if self._size < self._max_replay_buffer_size:
      self._size += 1

  def random_batch(self, batch_size, sample_key):
    assert batch_size % self.env_nums == 0, \
      "batch size should be dividable by env_nums"
    batch_size //= self.env_nums
    size = self.num_steps_can_sample()
    indices = np.random.randint(0, size, batch_size)
    return_dict = {}
    for key in sample_key:
      return_dict[key] = self.__getattribute__("_"+key)[indices]
      data_shape = (batch_size * self.env_nums,) + \
        return_dict[key].shape[2:]
      return_dict[key] = return_dict[key].reshape(data_shape)
    return return_dict

  def num_steps_can_sample(self):
    return self._size
