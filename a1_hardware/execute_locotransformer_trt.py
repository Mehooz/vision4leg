from control_loop_execution.rl_policy_wrapper import PolicyWrapper
from control_loop_execution.main_executor import Executor
from a1_utilities.robot_controller import RobotController
from a1_utilities.realsense import A1RealSense
from a1_utilities.a1_sensor_process import *
import pickle
from torchrl.utils import get_params
#from rich4loco.get_env import get_env

from control_loop_execution.trt_policy_wrapper import TRTPolicyWrapper

import glob
import os

if __name__ == "__main__":
  
  SEED_ = 2  
  POLICYNAME = "static-sim2sim-locotransformer-target0.6-thin-heightfield"
  trt_name = "locotransformer_0.6"
  EXECUTION_TIME = 15

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
  # params['net']['activation_func'] = torch.nn.ReLU
  obs_normalizer_mean = obs_normalizer._mean
  obs_normalizer_var = obs_normalizer._var
  get_image_interval = 4 #params['env']['env_build']['get_image_interval']
  num_action_repeat = params['env']['env_build']['num_action_repeat']

  PATH = "trt_engine/" + trt_name + "_seed" + str(SEED_) + "_fp16.trt"

  pf = TRTPolicyWrapper(
      PATH, 84, (4, 64, 64), 6, True
  )

  print("Warm Up")
  for i in range(100):
    f_input = np.random.rand(1, 84 + 64 *  64 *4)
    f_o = pf.eval_act(f_input)
  print("Warm Up doen")

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
