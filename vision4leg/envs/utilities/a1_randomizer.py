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

"""Abstract base class for environment randomizer."""


import abc
from .env_randomizer_base import EnvRandomizerBase
import numpy as np


class A1MassEnvRandomizer(EnvRandomizerBase):
  """Abstract base class for environment randomizer.

  Randomizes physical parameters of the objects in the simulation and adds
  perturbations to the stepping of the simulation.
  """

  def __init__(self, random_range):
    self.random_range = random_range
    self.env_info = np.array([1] * 16)
    self._name = "A1MassEnvRandomizer"
    self.lower_bound = np.array([1 - self.random_range] * 16)
    self.upper_bound = np.array([1 + self.random_range] * 16)

  def get_name(self):
    return self._name

  def change_mass(self, env, lid):
    scale = np.random.rand() * (2 * self.random_range) \
      - self.random_range + 1
    old_mass = env.robot._pybullet_client.getDynamicsInfo(
      env.robot.quadruped, lid)[0]
    env.robot._pybullet_client.changeDynamics(
      env.robot.quadruped, lid, mass=old_mass * scale)
    return scale - 1

  @abc.abstractmethod
  def randomize_env(self, env):
    """Randomize the simulated_objects in the environment.

    Will be called at when env is reset. The physical parameters will be fixed
    for that episode and be randomized again in the next environment.reset().

    Args:
    env: The Minitaur gym environment to be randomized.
    """
    self.env_info = []
    for lid in env.robot._hip_link_ids[1:]:
      self.env_info.append(self.change_mass(env, lid))

    for lid in env.robot._leg_link_ids:
      self.env_info.append(self.change_mass(env, lid))
    self.env_info = np.array(self.env_info)

  def randomize_step(self, env):
    """Randomize simulation steps.

    Will be called at every timestep. May add random forces/torques to Minitaur.

    Args:
    env: The Minitaur gym environment to be randomized.
    """
    pass
