
from control_loop_execution.rl_policy_wrapper import PolicyWrapper
from control_loop_execution.main_executor import Executor
from a1_utilities.robot_controller import RobotController
from a1_utilities.realsense import A1RealSense
from a1_utilities.a1_sensor_process import *
import pickle
import torch
import torchrl.networks.nets as networks
import torchrl.networks.base as networks_base
import torchrl.policies as policies
from torchrl.utils import get_params
from rich4loco.get_env import get_env

import glob
import os

if __name__ == "__main__":
  
  SEED_ = 0
  POLICYNAME = "locotransformer_target0.6_0.1_random_1.5_diagonal_z1_dense"
  EXECUTION_TIME = 8

  data_save_dir = "test_records/"

  if not os.path.exists(data_save_dir):
      os.makedirs(data_save_dir)

  idx = len(glob.glob(data_save_dir + "*"))

  print("Idx:",str(idx))

  comment = ""

  save_dir_name = data_save_dir + "%03d_%s_seed%d_%dseconds_%s/" % (idx, POLICYNAME, SEED_, int(EXECUTION_TIME) ,comment)

  if not os.path.exists(save_dir_name):
    os.makedirs(save_dir_name)

  robot_controller = RobotController(state_save_path=save_dir_name)
  realsense = A1RealSense(save_dir_name=save_dir_name)

  PARAM_PATH = "rl-policy/" + POLICYNAME + "/A1MoveGround/" + str(SEED_) + "/params.json"
  NORM_PATH = "rl-policy/" + POLICYNAME + "/A1MoveGround/" + str(SEED_) + "/model/_obs_normalizer_best.pkl"

  params = get_params(PARAM_PATH)
  with open(NORM_PATH, 'rb') as f:
    obs_normalizer = pickle.load(f)
  params['net']['activation_func'] = torch.nn.ReLU
  obs_normalizer_mean = obs_normalizer._mean
  obs_normalizer_var = obs_normalizer._var
  get_image_interval = 4 #params['env']['env_build']['get_image_interval']
  num_action_repeat = params['env']['env_build']['num_action_repeat']

  params['net']['base_type'] = networks.base.MLPBase
  encoder = networks.base.LocoTransformerEncoder(
      in_channels=4,
      state_input_dim=84,
      **params["encoder"]
  )

  pf = policies.GaussianContPolicyLocoTransformer(
      encoder=encoder,
      state_input_shape=84,
      visual_input_shape=(4, 64, 64),
      output_shape=6,
      **params["net"],
      **params["policy"]
  ).to("cuda:0")

  PATH = "rl-policy/" + POLICYNAME + "/A1MoveGround/" + str(SEED_) + "/model/model_pf_best.pth"

  current_pf_dict = pf.state_dict()
  current_pf_dict.update(torch.load(
      PATH,
      map_location="cuda:0")
  )

  pf.load_state_dict( 
      torch.load(
          PATH,
          map_location="cuda:0"
      )
  )

  print("Warm Up")
  for i in range(100):
    f_input = np.random.rand(1, 84 + 64 *  64 *4)
    f_o = pf.eval_act(f_input)
  print("Warm Up done")

  policyComputer = PolicyWrapper(
      pf, 
      obs_normalizer_mean, 
      obs_normalizer_var, 
      get_image_interval, 
      save_dir_name=save_dir_name,
      no_tensor=True,
      # default_joint_angle=[0, 0.67, -1.25]
      action_range=[0.05, 0.5, 0.5]
  )

  executor = Executor(
    realsense,
    robot_controller,
    policyComputer,
    control_freq = 25,
    frame_interval=get_image_interval,
    Kp=40, Kd=0.4
  )
  executor.execute(EXECUTION_TIME)
