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
"""This file implements the locomotion gym env."""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from vision4leg.envs.sensors import space_utils
import inspect
from absl import app
from absl import flags
import pybullet_data as pd
from pybullet_utils import bullet_client
import time
import pybullet
import random
from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
from mpc_controller import a1_sim as robot_sim
from copy import deepcopy
import vision4leg.envs.pybullet_client as bullet_client
import pybullet  # pytype: disable=import-error
import numpy as np
from gym.utils import seeding
from gym import spaces
import gym
import collections
import pkgutil
from collections import deque
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# import pybullet_utils.bullet_client as bullet_client


egl = pkgutil.get_loader('eglRenderer')

# from __future__ import google_type_annotations
currentdir = os.path.dirname(os.path.abspath(
  inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


# uncomment the robot of choice
# from mpc_controller import laikago_sim as robot_sim

FLAGS = flags.FLAGS


_NUM_SIMULATION_ITERATION_STEPS = 300


_STANCE_DURATION_SECONDS = [
  0.25
] * 4

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 50

_INIT_LEG_STATE = (
  gait_generator_lib.LegState.SWING,
  gait_generator_lib.LegState.STANCE,
  gait_generator_lib.LegState.STANCE,
  gait_generator_lib.LegState.SWING,
)


class LocomotionGymEnv(gym.Env):
  """The gym environment for the locomotion tasks."""
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 100
  }

  def __init__(self,
               gym_config,
               task=None,
               policy_freq=10,
               env_randomizers=None,
               empty_image=False,
               depth_image=False,
               depth_norm=False,
               get_image=False,
               vision_only=False,
               rgbd=False,
               grayscale=True,
               front=True,
               random_init_range=0,
               fric_coeff=[0.8, 0.1, 0.1],
               frame_extract=1,
               init_pos=None,
               init_ori=None,
               record_video=False,
               get_image_interval=1,
               reset_frame_idx=False,
               reset_frame_idx_each_step=False,
               blinding_spot=True,
               interpolation=False
               ):
    """Initializes the locomotion gym environment with an MPC controller as low level policy.

    Args:
      gym_config: An instance of LocomotionGymConfig.
      robot_class: A class of a robot. We provide a class rather than an
        instance due to hard_reset functionality. Parameters are expected to be
        configured with gin.
      sensors: A list of environmental sensors for observation.
      task: A callable function/class to calculate the reward and termination
        condition. Takes the gym env as the argument when calling.
      env_randomizers: A list of EnvRandomizer(s). An EnvRandomizer may
        randomize the physical property of minitaur, change the terrrain during
        reset(), or add perturbation forces during step().

    Raises:
      ValueError: If the num_action_repeat is less than 1.

    """
    self.count_t = 0
    self._policy_freq = policy_freq
    self.seed()
    self._gym_config = gym_config
    self.get_image_interval = get_image_interval
    self.init_pos = init_pos
    self.init_ori = init_ori
    self.vision_only = vision_only
    self.random_init_range = random_init_range
    self._num_simulation_iteration_steps = 30
    self.frame_extract = frame_extract
    self.reset_frame_idx = reset_frame_idx
    self.reset_frame_idx_each_step = reset_frame_idx_each_step
    assert frame_extract > 1 or not self.reset_frame_idx
    self.get_image = get_image
    self.empty_image = empty_image
    self.depth_image = depth_image
    self.depth_norm = depth_norm
    self.grayscale = grayscale
    self._record_video = record_video
    self.blinding_spot = blinding_spot
    self.interpolation = interpolation
    self.interpolation_delay = 3
    self.rgbd = rgbd
    self.feet_id = [2 + 3 * i for i in range(4)]
    if rgbd:
      if self.grayscale:
        self._image_channels = 8  # 2 * 4
      else:
        self._image_channels = 16  # (1 + 3) * 4
    else:
      if self.depth_image:
        self._image_channels = 4
      else:
        self._image_channels = 12
    self.front = front
    self.fric_coeff = fric_coeff
    self.stateId = -1
    self.reset_position_random_range = 0.3
    # A dictionary containing the objects in the world other than the robot.
    self._world_dict = {}
    self._task = task

    self._env_randomizers = env_randomizers if env_randomizers else []

    # This is a workaround due to the issue in b/130128505#comment5

    # Simulation related parameters.
    self._num_action_repeat = gym_config.simulation_parameters.num_action_repeat
    self._on_rack = gym_config.simulation_parameters.robot_on_rack
    if self._num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self._sim_time_step = gym_config.simulation_parameters.sim_time_step_s
    self._env_time_step = self._num_action_repeat * \
      self._sim_time_step * self._policy_freq
    self._env_step_counter = 0

    self._num_bullet_solver_iterations = int(
      self._num_simulation_iteration_steps / self._num_action_repeat)
    self._is_render = gym_config.simulation_parameters.enable_rendering

    # The wall-clock time at which the last frame is rendered.

    self.num_stored_frames = 4 * (self.frame_extract)
    # self.num_stored_frames = 4 + (self.frame_extract - 1) * 3
    self.frame_idx = [
      0,
      0 + self.frame_extract,
      0 + self.frame_extract * 2,
      0 + self.frame_extract * 3
    ]
    self.current_frames = deque(maxlen=self.num_stored_frames)
    self.depth_frames = deque(maxlen=self.num_stored_frames)

    if self._is_render:
      if self._record_video:
        self._pybullet_client = pybullet
        self._pybullet_client .connect(
          self._pybullet_client.GUI, options="--width=1280 --height=720 --mp4=\"test.mp4\" --mp4fps=100")
        self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
      else:
        self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
      # self._pybullet_client = bullet_client.BulletClient(
      #     connection_mode=pybullet.GUI)
      pybullet.configureDebugVisualizer(
        pybullet.COV_ENABLE_GUI,
        gym_config.simulation_parameters.enable_rendering_gui)
    else:
      self._pybullet_client = bullet_client.BulletClient(
        connection_mode=pybullet.DIRECT,
        # options=optionstring
      )
      if self.get_image:
        self.plugin_id = self._pybullet_client.loadPlugin(
          egl.get_filename(),
          "_eglRendererPlugin")
        # physicsClientId=self._pybullet_client._client)
        assert self.plugin_id != -1, 'Cannot load PyBullet plugin'
        print(self.plugin_id)

        self.pybullet_client.configureDebugVisualizer(
          pybullet.COV_ENABLE_RENDERING, 0)
        self.pybullet_client.configureDebugVisualizer(
          pybullet.COV_ENABLE_GUI, 0)

    self.pybullet_client.setAdditionalSearchPath(
      os.path.join(os.path.dirname(__file__), '../assets'))

    # if gym_config.simulation_parameters.egl_rendering:
    #   self._pybullet_client.loadPlugin('eglRendererPlugin')

    # The action list contains the name of all actions.
    self._build_action_space()

    # Set the default render options.
    self._camera_dist = gym_config.simulation_parameters.camera_distance
    self._camera_yaw = gym_config.simulation_parameters.camera_yaw
    self._camera_pitch = gym_config.simulation_parameters.camera_pitch
    self._render_width = gym_config.simulation_parameters.render_width
    self._render_height = gym_config.simulation_parameters.render_height

    self._hard_reset = True
    self.reset()

    self._hard_reset = gym_config.simulation_parameters.enable_hard_reset

    # Construct the observation space from the list of sensors. Note that we
    # will reconstruct the observation_space after the robot is created.
    gym_space_dict = {}
    # if not self.vision_only:
    gym_space_dict["com_vel"] = spaces.Box(np.array([-2, -2, -2]),
                                           np.array([2, 2, 2]),
                                           dtype=np.float32)
    gym_space_dict["imu"] = spaces.Box(np.array([-2.0 * np.pi] * 3),
                                       np.array([2.0 * np.pi] * 3),
                                       dtype=np.float32)
    self.observation_space = (spaces.Dict(gym_space_dict))
    self.augment_observation_space_with_randomizer()

  def augment_observation_space_with_randomizer(self):
    # print(self.observation_space)
    # print()
    self.model_condition_dim = 0
    self.observation_space = self.observation_space.spaces
    for r in self._env_randomizers:
      if hasattr(r, 'upper_bound'):
        self.observation_space[r.get_name()] = spaces.Box(
          r.upper_bound, r.lower_bound
        )
        self.model_condition_dim += len(r.upper_bound)
    self.observation_space = spaces.Dict(self.observation_space)

  def _setup_controller(self):
    """Demonstrates how to create a locomotion controller."""
    self.desired_speed = (0, 0)
    self.desired_twisting_speed = 0

    self.gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      self._robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
    self.state_estimator = com_velocity_estimator.COMVelocityEstimator(self._robot,
                                                                       window_size=20)
    self.sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      self._robot,
      self.gait_generator,
      self.state_estimator,
      desired_speed=self.desired_speed,
      desired_twisting_speed=self.desired_twisting_speed,
      desired_height=robot_sim.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

    self.st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      self._robot,
      self.gait_generator,
      self.state_estimator,
      desired_speed=self.desired_speed,
      desired_twisting_speed=self.desired_twisting_speed,
      desired_body_height=robot_sim.MPC_BODY_HEIGHT,
      body_mass=robot_sim.MPC_BODY_MASS,
      body_inertia=robot_sim.MPC_BODY_INERTIA)

    self.controller = locomotion_controller.LocomotionController(
      self._robot,
      gait_generator=self.gait_generator,
      state_estimator=self.state_estimator,
      swing_leg_controller=self.sw_controller,
      stance_leg_controller=self.st_controller,
      clock=self._robot.GetTimeSinceReset)

    self.controller.reset()

  def _update_controller_params(self, lin_speed, ang_speed):
    self.controller.swing_leg_controller.desired_speed = lin_speed
    self.controller.swing_leg_controller.desired_twisting_speed = ang_speed
    self.controller.stance_leg_controller.desired_speed = lin_speed
    self.controller.stance_leg_controller.desired_twisting_speed = ang_speed

  def _build_action_space(self):
    """Builds action space based on target CoM vel."""
    self.action_space = spaces.Box(np.array([-0.4, -0.4]),
                                   np.array([0.4, 0.4]),
                                   dtype=np.float32)

  def close(self):
    if hasattr(self, '_robot') and self._robot:
      self._robot.Terminate()

  def seed(self, seed=None):
    self.np_random, self.np_random_seed = seeding.np_random(seed)
    return [self.np_random_seed]

  def reset(self,
            initial_motor_angles=None,
            reset_duration=0.0,
            reset_visualization_camera=True):
    """Resets the robot's position in the world or rebuild the sim world.

    The simulation world will be rebuilt if self._hard_reset is True.

    Args:
      initial_motor_angles: A list of Floats. The desired joint angles after
        reset. If None, the robot will use its built-in value.
      reset_duration: Float. The time (in seconds) needed to rotate all motors
        to the desired initial values.
      reset_visualization_camera: Whether to reset debug visualization camera on
        reset.

    Returns:
      A numpy array contains the initial observation after reset.
    """
    del initial_motor_angles
    del reset_duration

    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
        self._pybullet_client.COV_ENABLE_RENDERING, 0)
    if self.reset_frame_idx:
      assert self.frame_extract > 1
      rand_indices = self.np_random.randint(0, self.frame_extract, 4)
      self.frame_idx = [
        rand_indices[0],
        rand_indices[1] + self.frame_extract,
        rand_indices[2] + self.frame_extract * 2,
        rand_indices[3] + self.frame_extract * 3
      ]
    if self.interpolation:
      self.interpolation_delay = np.random.randint(0, self.frame_extract)
    # Clear the simulation world and rebuild the robot interface.
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      # print("="*30, self._num_simulation_iteration_steps)
      self._pybullet_client.setPhysicsEngineParameter(
        numSolverIterations=self._num_bullet_solver_iterations)
      self._pybullet_client.setTimeStep(self._sim_time_step)
      self._pybullet_client.setGravity(0, 0, -9.8)
      robot_uid = self.pybullet_client.loadURDF(
        robot_sim.URDF_NAME, robot_sim.START_POS)
      self._robot = robot_sim.SimpleRobot(
        self.pybullet_client, robot_uid, simulation_time_step=self._sim_time_step)
      self._last_base_orientation = self.init_ori = self._robot._GetDefaultInitOrientation()
      # Loop over all env randomizers.
      self._setup_controller()
      # Rebuild the world.
      self._world_dict = {
        "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
      }

      # original_path = pybullet_data.getDataPath()
      # self.pybullet_client.setAdditionalSearchPath(os.path.dirname(__file__))
    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_env(self)
    # Reset the pose of the robot.
    self._robot.ResetPose()

    init_pos = deepcopy(self.init_pos)
    init_pos[0] += (np.random.rand() * self.reset_position_random_range *
                    2 - self.reset_position_random_range)
    init_pos[1] += (np.random.rand() * self.reset_position_random_range *
                    2 - self.reset_position_random_range)
    self._pybullet_client.resetBasePositionAndOrientation(
      self._robot.quadruped, self.init_pos, self.init_ori)
    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self._env_step_counter = 0
    if reset_visualization_camera:
      self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist,
                                                       self._camera_yaw,
                                                       self._camera_pitch,
                                                       [0, 0, 0])
    self._last_action = np.zeros(self.action_space.shape)

    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
        self._pybullet_client.COV_ENABLE_RENDERING, 1)

    if self._task and hasattr(self._task, 'reset'):
      self._task.reset(self)

    self.pybullet_client.changeDynamics(
      self._world_dict["ground"], -1,
      lateralFriction=self.fric_coeff[0],
      spinningFriction=self.fric_coeff[1],
      rollingFriction=self.fric_coeff[2]
    )
    self.controller.reset()
    return self._get_observation(reset=True)

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: Can be a list of desired motor angles for all motors when the
        robot is in position control mode; A list of desired motor torques. Or a
        list of tuples (q, qdot, kp, kd, tau) for hybrid control mode. The
        action must be compatible with the robot's motor control mode. Also, we
        are not going to use the leg space (swing/extension) definition at the
        gym level, since they are specific to Minitaur.

    Returns:
      observations: The observation dictionary. The keys are the sensor names
        and the values are the sensor readings.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    self._last_base_position = self._robot.GetBasePosition()
    self._last_base_orientation = self._robot.GetBaseOrientation()
    self._last_action = action

    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
        self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_step(self)
    # control xy
    lin_speed, ang_speed = action[0], action[1]
    lin_speed = np.concatenate([[lin_speed], np.zeros(2)])
    lin_speed = np.clip(lin_speed, a_min=-0.05, a_max=None)

    self._update_controller_params(lin_speed, ang_speed)

    for _ in range(self._policy_freq):
      self.controller.update()
      action, _ = self.controller.get_action()
      self._robot.Step(action)

    if self._task and hasattr(self._task, 'update'):
      self._task.update(self)

    reward = self._reward()

    done = self._termination()
    self._env_step_counter += 1
    if done:
      self._robot.Terminate()
    return self._get_observation(), reward, done, {}

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError('Unsupported render mode:{}'.format(mode))
    base_pos = self._robot.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
      cameraTargetPosition=base_pos,
      distance=self._camera_dist,
      yaw=self._camera_yaw,
      pitch=self._camera_pitch,
      roll=0,
      upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
      fov=60,
      aspect=float(self._render_width) / self._render_height,
      nearVal=0.1,
      farVal=100.0)
    (_, _, px, _, _) = self.pybullet_client.getCameraImage(
      width=self._render_width,
      height=self._render_height,
      renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
      viewMatrix=view_matrix,
      projectionMatrix=proj_matrix)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_ground(self):
    """Get simulation ground model."""
    return self._world_dict['ground']

  @ property
  def ground_id(self):
    return self._world_dict['ground']

  @ ground_id.setter
  def ground_id(self, id):
    self._world_dict['ground'] = id

  def set_ground(self, ground_id):
    """Set simulation ground model."""
    self._world_dict['ground'] = ground_id

  @ property
  def rendering_enabled(self):
    return self._is_render

  @ property
  def last_base_position(self):
    return self._last_base_position

  @ property
  def world_dict(self):
    return self._world_dict.copy()

  @ world_dict.setter
  def world_dict(self, new_dict):
    self._world_dict = new_dict.copy()

  def _termination(self):
    if not self._robot.is_safe:
      return True
    if self._task and hasattr(self._task, 'done'):
      return self._task.done(self)

    for s in self.all_sensors():
      s.on_terminate(self)
    return False

  def _reward(self):
    if self._task:
      return self._task(self)
    return 0

  def _get_observation(self, reset=False):
    """Get observation of this environment from a list of sensors.

    Returns:
      observations: sensory observation in the numpy array format
    """
    sensors_dict = {}
    if not self.vision_only:
      sensors_dict = {
        "com_vel": self.state_estimator._com_velocity_world_frame,
        "imu": self._robot.GetBaseRollPitchYaw()}

    observations = collections.OrderedDict(
      sorted(list(sensors_dict.items())))
    if self.get_image and self._env_step_counter % self.get_image_interval == 0:
      if self.reset_frame_idx_each_step:
        # assert self.frame_extract > 1
        self.frame_idx = [
          self.np_random.randint(1, self.frame_extract)
        ] + [self.frame_idx[i] + self.frame_extract for i in range(3)]
      if self.empty_image:
        grayscale = np.zeros((self._image_channels, 64, 64))
        if reset:
          for i in range(self.num_stored_frames):
            self.current_frames.appendleft(grayscale)
        else:
          self.current_frames.appendleft(grayscale)

        observations['raw_img'] = np.concatenate(
          [self.current_frames[idx] for idx in self.frame_idx],
          axis=0
        ).reshape(-1)
        return observations

      linkstate = self.pybullet_client.getLinkState(
        self._robot.quadruped, 0, computeForwardKinematics=True)
      camInfo = self.pybullet_client.getDebugVisualizerCamera()
      proj_mat = camInfo[3]
      proj_mat = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
      ]
      camOrn = linkstate[1]
      camMat = self.pybullet_client.getMatrixFromQuaternion(camOrn)
      forwardVec = [camMat[0], camMat[3],
                    camMat[6]]
      camPos = linkstate[0]
      camPos = [camPos[i] + forwardVec[i] * 0.2309 for i in range(3)]
      self.camPos = camPos

      camUpVec2 = [
        (camMat[2] + camMat[0]) / 2,
        (camMat[5] + camMat[3]) / 2,
        (camMat[8] + camMat[6]) / 2]
      forwardVec2 = [
        (-camMat[2] + camMat[0]) / 2,
        (-camMat[5] + camMat[3]) / 2,
        (-camMat[8] + camMat[6]) / 2]

      if self.front:
        camUpVec2 = [
          camMat[2],
          camMat[5],
          camMat[8]]
        forwardVec2 = [
          camMat[0],
          camMat[3],
          camMat[6]]
        camUpVec2 = [0, 0, 1]
        forwardVec[2] = 0

      camTarget2 = [camPos[i] + forwardVec2[i] * 10 for i in range(3)]
      viewMat2 = self.pybullet_client.computeViewMatrix(
        camPos, camTarget2, camUpVec2
      )

      camera_image_set = self.pybullet_client.getCameraImage(
        64, 64, viewMatrix=viewMat2, projectionMatrix=proj_mat,
        # flags=pybullet.ER_NO_SEGMENTATION_MASK,
        shadow=1,
        lightDirection=[1, 1, 1],
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
      )
      _, _, _, depth_img, _ = camera_image_set

      depth = depth_img[np.newaxis, ...]
      if self.depth_image:
        far = 1000
        near = 0.01
        depth = far * near / (far - (far - near) * depth)
        if self.blinding_spot:
          num_spots = self.np_random.randint(3, 30)
          blinding_indices = self.np_random.randint(
            0, 64, size=(num_spots, 2))
          for idx in blinding_indices:
            depth[:, idx[0], idx[1]] = 10
        depth = np.clip(depth, a_min=0.3, a_max=10)
        depth = np.sqrt(np.log(depth + 1))

      if reset:
        for _ in range(self.num_stored_frames):
          self.depth_frames.appendleft(depth)
      else:
        self.depth_frames.appendleft(depth)
    if self.get_image:
      if self.interpolation:
        concated_depths = []
        for idx in self.frame_idx:
          img = self.depth_frames[idx].copy()
          for k in range(self.interpolation_delay):
            img += self.depth_frames[idx + k + 1]
          concated_depths.append(
            img / (self.interpolation_delay + 1))
          # print(concated_depths)
        concated_depths = np.concatenate(
          concated_depths, axis=0).reshape(-1)
      else:
        concated_depths = np.concatenate(
          [self.depth_frames[idx] for idx in self.frame_idx],
          axis=0
        ).reshape(-1)
      if self.depth_norm and self.depth_image:
        concated_depths = (concated_depths - 1.25) / 0.425

      if self.rgbd:
        raise NotImplementedError
      else:
        if self.depth_image:
          observations['raw_img'] = concated_depths
    return observations

  def set_time_step(self, num_action_repeat, sim_step=0.001):
    """Sets the time step of the environment.

    Args:
      num_action_repeat: The number of simulation steps/action repeats to be
        executed when calling env.step().
      sim_step: The simulation time step in PyBullet. By default, the simulation
        step is 0.001s, which is a good trade-off between simulation speed and
        accuracy.

    Raises:
      ValueError: If the num_action_repeat is less than 1.
    """
    if num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self._sim_time_step = sim_step
    self._num_action_repeat = num_action_repeat
    self._env_time_step = sim_step * num_action_repeat
    self._num_bullet_solver_iterations = (
      self._num_simulation_iteration_steps / self._num_action_repeat)
    self._pybullet_client.setPhysicsEngineParameter(
      numSolverIterations=int(np.round(self._num_bullet_solver_iterations)))
    self._pybullet_client.setTimeStep(self._sim_time_step)
    self._robot.SetTimeSteps(self._num_action_repeat, self._sim_time_step)

  def get_time_since_reset(self):
    """Get the time passed (in seconds) since the last reset.

    Returns:
      Time in seconds since the last reset.
    """
    return self._robot.GetTimeSinceReset()

  @ property
  def pybullet_client(self):
    return self._pybullet_client

  @ property
  def robot(self):
    return self._robot

  @ property
  def env_step_counter(self):
    return self._env_step_counter

  @ property
  def hard_reset(self):
    return self._hard_reset

  @ property
  def last_action(self):
    return self._last_action

  @ property
  def env_time_step(self):
    return self._env_time_step

  @ property
  def task(self):
    return self._task

  @ property
  def robot_class(self):
    return self._robot_class

  @ property
  def image_channels(self):
    return self._image_channels

  @ property
  def world_dict(self):
    return self._world_dict
