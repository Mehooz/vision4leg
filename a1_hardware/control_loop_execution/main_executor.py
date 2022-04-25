import threading
import time
import numpy as np
from a1_utilities.realsense import A1RealSense
from a1_utilities.a1_sensor_process import *
from a1_utilities.InformationSaver import InformationSaver
from a1_utilities.predefined_pose import move_to_sit, move_to_stand


class Executor:
  def __init__(
    self,
    realsense_device,
    robot_controller,
    policy,
    use_high_command = False,
    control_freq = 50, frame_interval = 4, Kp = 40, Kd = 0.4
  ):
    self.realsense_device = realsense_device
    self.robot_controller = robot_controller
    self.policy = policy
    self.continue_thread = False
    self.control_freq = control_freq
    self.frame_extract = frame_interval

    self.Kp = Kp
    self.Kd = Kd

    self.execution_thread = None

    self.use_high_command = use_high_command

  def warmup_observations(self):
    self.last_action = np.zeros(12)
    # Fill sensor history buffer with observation
    for i in range(3):  # sensor history buffer have length 3 at most
      observation = self.robot_controller.get_observation()
      print(observation)
      if not self.policy.vis_only:
        # fill IMU and joint angle sensor buffer
        # IMU
        IMU_hist_normalized = self.policy.imu_historical_data.record_and_normalize(
            np.array([
                observation.imu.rpy[0],
                observation.imu.rpy[1],
                observation.imu.gyroscope[0],
                observation.imu.gyroscope[1],
            ])# R, P, dR, dP; dR,dP are not literally speed of roll, but angular velocity on corresponding axis in body frame. 
        )
        # joint angle
        joint_angle_hist_normalized = self.policy.joint_angle_historical_data.record_and_normalize(
            observation_to_joint_position(observation)
        )

        if self.policy.use_foot_contact:
          foot_contact_normalized = self.policy.foot_contact_historical_data.record_and_normalize(
            np.array(observation.footForce) > 20
          )
      time.sleep(0.05)

    # read one frame
    self.depth_scale, self.curr_frame = self.realsense_device.get_depth_frame()
    for i in range(self.frame_extract*3+1):
        # fill action
        action = self.policy.get_action(
          self.robot_controller.get_observation(), self.curr_frame, self.depth_scale,
          self.last_action
        ) # force frame update
        
        last_action_normalized = self.policy.last_action_historical_data.record_and_normalize(
          self.last_action
        )
        self.last_action = action 
        time.sleep(0.05)
    print("Policy thread initialization done!")

  def main_execution(self):
    count = 0
    if hasattr(self.policy.pf, "cuda_cxt"):
      self.policy.pf.cuda_cxt.push()

    while self.continue_thread:
      start_time = time.time()
      # Get observation
      robot_observation = self.robot_controller.get_observation()
      # Get frame every time
      depth_scale, curr_frame = self.realsense_device.get_depth_frame()
      # compute next action
      action = self.policy.get_action(
        robot_observation, curr_frame, depth_scale,
        self.last_action
      )
      print(action)
      self.last_action = action

      # prepare command
      if self.use_high_command:
        command = prepare_high_level_cmd(action)
      else:
        command = prepare_position_cmd(action, self.Kp, self.Kd)
      self.robot_controller.set_action(command)

      end_time = time.time()
      # control loop frequency
      count += 1

      delay = end_time - start_time
      delay_time = 1 / self.control_freq - delay
      time.sleep(max(0, delay_time))

    if hasattr(self.policy.pf, "cuda_cxt"):
      self.policy.pf.cuda_cxt.pop()

  def start_thread(self):
    print("Start policy thread called")
    self.continue_thread = True
    self.execution_thread = threading.Thread(target=self.main_execution)
    self.execution_thread.start()

  def stop_thread(self):
    print("Stop policy thread called")
    self.continue_thread = False
    # self.policy.write()
    self.execution_thread.join()

  def execute(self, execution_time):
    self.realsense_device.start_thread()
    self.robot_controller.start_thread()
    time.sleep(1)
    self.warmup_observations()

    move_to_stand(self.robot_controller)
    self.start_thread()
    time.sleep(execution_time) # RUN POLICY FOR TEN SECONDS?
    self.stop_thread()
    # Get robot to sitting position
    move_to_sit(self.robot_controller)
    # Terminate all processes
    self.realsense_device.stop_thread()
    self.robot_controller.stop_thread()
    self.policy.write()
