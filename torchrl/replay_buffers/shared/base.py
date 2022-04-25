# Since we could ensure that multi-proces would write into the different parts
# For efficiency, we use Multiprocess.RawArray
from torch.multiprocessing import RawArray
from multiprocessing.managers import BaseProxy
import numpy as np
from torchrl.replay_buffers.base import BaseReplayBuffer
from .shmarray import NpShmemArray
from .shmarray import get_random_tag


class SharedBaseReplayBuffer(BaseReplayBuffer):
  """
  Basic Replay Buffer
  """

  def __init__(
      self, max_replay_buffer_size, worker_nums):
    super().__init__(max_replay_buffer_size)

    self.worker_nums = worker_nums
    assert self._max_replay_buffer_size % self.worker_nums == 0, \
      "buffer size is not dividable by worker num"
    self._max_replay_buffer_size //= self.worker_nums

    if not hasattr(self, "tag"):
      self.tag = get_random_tag()

  def build_by_example(self, example_dict):
    self._size = NpShmemArray(self.worker_nums, np.int32, self.tag+"_size")
    self._top = NpShmemArray(self.worker_nums, np.int32, self.tag+"_top")

    self.tags = {}
    self.shapes = {}
    for key in example_dict:
      if not hasattr(self, "_" + key):
        current_tag = "_"+key
        self.tags[current_tag] = self.tag+current_tag
        shape = (self._max_replay_buffer_size, self.worker_nums) + \
          np.shape(example_dict[key])
        self.shapes[current_tag] = shape

        np_array = NpShmemArray(
          shape, np.float32, self.tag+current_tag)
        self.__setattr__(current_tag, np_array)

  def rebuild_from_tag(self):
    self._size = NpShmemArray(
      self.worker_nums, np.int32,
      self.tag+"_size", create=False)
    self._top = NpShmemArray(
      self.worker_nums, np.int32,
      self.tag+"_top", create=False)

    for key in self.tags:
      np_array = NpShmemArray(
        self.shapes[key], np.float32,
        self.tags[key], create=False)
      self.__setattr__(key, np_array)

  def add_sample(self, sample_dict, worker_rank, **kwargs):
    for key in sample_dict:
      self.__getattribute__("_" + key)[
        self._top[worker_rank], worker_rank] = \
        sample_dict[key]
    self._advance(worker_rank)

  def terminate_episode(self):
    pass

  def _advance(self, worker_rank):
    self._top[worker_rank] = (self._top[worker_rank] + 1) % \
      self._max_replay_buffer_size
    if self._size[worker_rank] < self._max_replay_buffer_size:
      self._size[worker_rank] = self._size[worker_rank] + 1

  def random_batch(self, batch_size, sample_key):
    assert batch_size % self.worker_nums == 0, \
      "batch size should be dividable by worker_nums"
    batch_size //= self.worker_nums
    size = self.num_steps_can_sample()
    indices = np.random.randint(0, size, batch_size)
    return_dict = {}
    for key in sample_key:
      return_dict[key] = self.__getattribute__("_"+key)[indices].reshape(
        (batch_size * self.worker_nums, -1))
    return return_dict

  def num_steps_can_sample(self):
    min_size = np.min(self._size)
    max_size = np.max(self._size)
    assert max_size == min_size, \
      "all worker should gather the same amount of samples"
    return min_size


class AsyncSharedReplayBuffer(SharedBaseReplayBuffer):
  def num_steps_can_sample(self):
    # Use asynchronized sampling could cause sample collected is
    # different across different workers but actually it's find
    min_size = np.min(self._size)
    return min_size
