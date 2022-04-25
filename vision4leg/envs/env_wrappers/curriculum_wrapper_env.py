# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A wrapper for motion imitation environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


class CurriculumWrapperEnv(object):
  """An env using for training policy with motion imitation."""

  def __init__(self,
               gym_env,
               episode_length_start=1000,
               episode_length_end=2000,
               curriculum_steps=10000000,
               num_parallel_envs=1):
    """Initialzes the wrapped env.

    Args:
      gym_env: An instance of LocomotionGymEnv.
    """
    self._gym_env = gym_env
    self.observation_space = self._gym_env.observation_space

    self._episode_length_start = episode_length_start
    self._episode_length_end = episode_length_end
    self._curriculum_steps = int(
      np.ceil(curriculum_steps / num_parallel_envs))
    self._total_step_count = 0

    if self._enable_curriculum():
      self._update_time_limit()

    self.seed()
    return

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)

  def step(self, action):
    """Steps the wrapped environment.

    Args:
      action: Numpy array. The input action from an NN agent.

    Returns:
      The tuple containing the modified observation, the reward, the epsiode end
      indicator.

    Raises:
      ValueError if input action is None.

    """
    self._total_step_count += 1
    return self._gym_env.step(action)

  def reset(self):
    if self._enable_curriculum():
      self._update_time_limit()
    return self._gym_env.reset()

  def _enable_curriculum(self):
    """Check if curriculum is enabled."""
    return self._curriculum_steps > 0

  def _update_time_limit(self):
    """Updates the current episode length depending on the number of environment steps taken so far."""
    t = float(self._total_step_count) / self._curriculum_steps
    t = np.clip(t, 0.0, 1.0)
    t = np.power(t, 3.0)
    new_steps = int((1.0 - t) * self._episode_length_start +
                    t * self._episode_length_end)
    self._max_episode_steps = new_steps
    return
