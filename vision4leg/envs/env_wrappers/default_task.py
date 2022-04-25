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

"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class DefaultTask(object):
  """Default empy task."""

  def __init__(self):
    """Initializes the task."""
    self._draw_ref_model_alpha = 1.
    self._ref_model = -1
    return

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    return

  def update(self, env):
    """Updates the internal state of the task."""
    del env
    return

  def done(self, env):
    """Checks if the episode is over."""
    del env
    return False

  def reward(self, env):
    """Get the reward without side effects."""
    del env
    return self._calc_reward_root_velocity()
    # return 1

  def _get_pybullet_client(self):
    """Get bullet client from the environment"""
    return self._env._pybullet_client

  def _calc_reward_root_velocity(self):
    """Get the root velocity reward."""
    env = self._env
    robot = env.robot
    sim_model = robot.quadruped
    # ref_model = self._ref_model
    pyb = self._get_pybullet_client()

    # root_vel_ref, root_ang_vel_ref = pyb.getBaseVelocity(ref_model)
    root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)
    # root_vel_ref = np.array(root_vel_ref)
    # root_ang_vel_ref = np.array(root_ang_vel_ref)
    root_vel_sim = np.array(root_vel_sim)
    # print(root_vel_sim)
    # root_ang_vel_sim = np.array(root_ang_vel_sim)

    # root_vel_diff = root_vel_ref - root_vel_sim
    # root_vel_err = root_vel_diff.dot(root_vel_diff)

    # root_ang_vel_diff = root_ang_vel_ref - root_ang_vel_sim
    # root_ang_vel_err = root_ang_vel_diff.dot(root_ang_vel_diff)

    # root_velocity_err = root_vel_err + 0.1 * root_ang_vel_err
    # root_velocity_reward = np.exp(-self._root_velocity_err_scale *
    # root_velocity_err)

    return root_vel_sim[0]
