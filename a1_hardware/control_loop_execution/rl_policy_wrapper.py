from a1_utilities.a1_sensor_histories import NormedStateHistory
from a1_utilities.a1_sensor_process import observation_to_joint_position, observation_to_torque
from a1_utilities.logger import StateLogger
from a1_utilities.a1_sensor_histories import VisualHistory
import numpy as np


class PolicyWrapper():
  def __init__(
    self,
    policy,
    obs_normalizer_mean, obs_normalizer_var,
    get_image_interval,
    save_dir_name, 
    sliding_frames=True, no_tensor=False, 
    default_joint_angle=None,
    action_range=None,
    vis_only=False,
    state_only=False,
    clip_motor=False,
    clip_motor_value=0.5,
    use_foot_contact=False,
    save_log=False
  ):
    self.pf = policy
    self.no_tensor = no_tensor

    self.get_image_interval = get_image_interval

    self.vis_only = vis_only
    self.state_only = state_only

    if default_joint_angle == None:
      default_joint_angle = [0.0, 0.9,-1.8]
    self.default_joint_angle = np.array(default_joint_angle * 4)

    self.current_joint_angle = default_joint_angle
    self.clip_motor = clip_motor
    self.clip_motor_value = clip_motor_value
    if action_range == None:
      action_range = [0.05, 0.5, 0.5]
    self.action_range = np.array(action_range * 4)

    self.action_lb = self.default_joint_angle - self.action_range
    self.action_ub = self.default_joint_angle + self.action_range

    if not self.vis_only:
      self.use_foot_contact = use_foot_contact
      last_start=0
      if use_foot_contact:
        self.foot_contact_historical_data = NormedStateHistory(
          input_dim=4,
          num_hist=3,
          mean=obs_normalizer_mean[0:12],
          var=obs_normalizer_var[0:12]
        )
        last_start = 12

      self.imu_historical_data = NormedStateHistory(
        input_dim=4,
        num_hist=3,
        mean=obs_normalizer_mean[last_start: last_start+12],
        var=obs_normalizer_var[last_start: last_start+12]
      )

      self.last_action_historical_data = NormedStateHistory(
        input_dim=12,
        num_hist=3,
        mean=obs_normalizer_mean[last_start + 48: last_start + 84],
        var=obs_normalizer_var[last_start + 48: last_start + 84]
      )

      self.joint_angle_historical_data = NormedStateHistory(
        input_dim=12,
        num_hist=3,
        mean=obs_normalizer_mean[last_start + 12 : last_start + 48],
        var=obs_normalizer_var[last_start + 12 : last_start + 48]
      )

    self.frames_historical_data = VisualHistory(
      frame_shape = (64,64),
      num_hist = get_image_interval * 3 + 1,
      mean = 1.25 * np.ones(
        64 * 64 * (get_image_interval * 3 + 1)
      ),   # FIXME: Mean measured = 1.02
      var = 0.425 ** 2 * np.ones(
        64 * 64 * (get_image_interval * 3 + 1)
      ),    # FIXME: Variance measured = 0.11
      sliding_frames=sliding_frames
    ) # variance, not std!

    self.save_log = save_log
    if self.save_log:
      # array savers
      self.ob_tensor_saver = StateLogger(
        np.zeros((16468),dtype=np.float32),
        duration=60, 
        frequency=25, 
        data_save_name=save_dir_name + "ob_t.npz"
      )

      self.policy_action_saver = StateLogger(
        np.zeros((12),dtype=np.float32),
        duration=60, 
        frequency=25, 
        data_save_name=save_dir_name + "policy_action.npz"
      )

  def process_obs(self, observation, depth_frame, depth_scale, last_action):
    if not self.vis_only:
      # IMU
      imu_hist_normalized = self.imu_historical_data.record_and_normalize(
        np.array([
            observation.imu.rpy[0],
            observation.imu.rpy[1],
            observation.imu.gyroscope[0],
            observation.imu.gyroscope[1],
        ])# R, P, dR, dP; dR,dP are not literally speed of roll, but angular velocity on corresponding axis in body frame. 
      )

      # joint angle
      joint_angle = observation_to_joint_position(observation)
      self.current_joint_angle = joint_angle
      joint_angle_hist_normalized = self.joint_angle_historical_data.record_and_normalize(
        joint_angle
      )

      # last action
      last_action_normalized = self.last_action_historical_data.record_and_normalize(
        last_action
      )
      if self.use_foot_contact:
        foot_contact_normalized = self.foot_contact_historical_data.record_and_normalize(
          np.array(observation.footForce) > 20
        )
      # append new frame everytime and give index 0,4,8,12
      normalized_visual_history = self.frames_historical_data.record_and_normalize(depth_frame, depth_scale, backwards=True)

    ## concatnate all observations and feed into network
    if self.vis_only:
      obs_normalized_np = normalized_visual_history.reshape(-1)
    elif self.state_only:
      obs_list = []
      if self.use_foot_contact:
        obs_list.append(foot_contact_normalized)
      obs_list += [
        imu_hist_normalized,
        last_action_normalized, 
        joint_angle_hist_normalized,
      ]
      obs_normalized_np = np.hstack(obs_list)
    else:
      obs_list = []
      if self.use_foot_contact:
        obs_list.append(foot_contact_normalized)
      obs_list += [
        imu_hist_normalized,
        last_action_normalized, 
        joint_angle_hist_normalized,
        normalized_visual_history.reshape(-1)
      ]
      obs_normalized_np = np.hstack(obs_list)

    if self.save_log:
      self.ob_tensor_saver.record(obs_normalized_np)

    if not self.no_tensor:
      import torch
      ob_t = torch.Tensor(obs_normalized_np).unsqueeze(0).to("cuda:0")
    else:
      ob_t = obs_normalized_np[np.newaxis,:]
    return ob_t

  def process_act(self, action):
    if self.vis_only:
      return action
    else:
      diagonal_action_normalized = action
      right_act_normalized, left_act_normalized = np.split(diagonal_action_normalized, 2)
      action_normalized = np.concatenate(
          [right_act_normalized, left_act_normalized, left_act_normalized, right_act_normalized]
      )

      action_ub = self.action_ub
      action_lb = self.action_lb
      action = 0.5 * (np.tanh(action_normalized) + 1) * (action_ub - action_lb) + action_lb
      if self.clip_motor:
        action = np.clip(
          action,
          self.current_joint_angle - self.clip_motor_value,
          self.current_joint_angle + self.clip_motor_value
        )
    return action


  def get_action(self, observation, depth_frame, depth_scale, last_action):
    '''
    This function process raw observation, fed normalized observation into
    the network, de-normalize and output the action.
    '''
    ob_t = self.process_obs(observation, depth_frame, depth_scale, last_action)
    action = self.pf.eval_act(ob_t)
    action = self.process_act(action)
    if self.save_log:
      self.policy_action_saver.record(action)
    return action

  def write(self):
    if self.save_log:
      self.ob_tensor_saver.write()
      self.policy_action_saver.write()
