import sys
sys.path.append(".")
from pathlib import Path
import cv2
import pickle
import pybullet
import numpy as np
import time
import matplotlib.pyplot as plt
import torchrl.networks as networks
import torchrl.policies as policies
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import gym.wrappers as wrappers
import gym
import torch.nn.functional as F
from torchrl.utils import get_params
from vision4leg.get_env import get_env
import seaborn as sns
import random
import argparse
import torch
import os


def get_args():
  parser = argparse.ArgumentParser(description='RL')
  parser.add_argument('--seed', type=int, default=0,
                      help='random seed (default: 1)')
  parser.add_argument('--env_seed', type=int, default=0,
                      help='random seed (default: 1)')
  parser.add_argument('--env_name', type=str, default='A1MoveForwardBurden')
  parser.add_argument('--num_episodes', type=int, default=1,
                      help='number of episodes')
  parser.add_argument("--config", type=str,   default=None,
                      help="config file", )
  parser.add_argument('--save_dir', type=str, default='./snapshots',
                      help='directory for snapshots (default: ./snapshots)')
  parser.add_argument('--video_dir', type=str, default='./video')
  parser.add_argument('--log_dir', type=str, default='./log',
                      help='directory for tensorboard logs (default: ./log)')
  parser.add_argument('--no_cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument("--device", type=int, default=0,
                      help="gpu secification", )
  parser.add_argument('--add_tag', type=str, default='',
                      help='directory for snapshots (default: ./snapshots)')
  parser.add_argument('--snap_check', type=str, default='best')

  # tensorboard
  parser.add_argument("--id", type=str,   default=None,
                      help="id for tensorboard", )

  args = parser.parse_args()

  args.cuda = not args.no_cuda and torch.cuda.is_available()

  return args


args = get_args()

# params = get_params(args.config)


# from metaworld_utils.meta_env import get_meta_env


np.random.seed(0)
random.seed(0)

PARAM_PATH = "{}/{}/{}/{}/params.json".format(
  args.log_dir,
  args.id,
  args.env_name,
  args.seed
)
# params = get_params(args.config)
params = get_params(PARAM_PATH)

# params["env"]["env_build"]["enable_rendering"] = False
params["env"]["env_build"]["enable_rendering"] = True
params["env"]["env_build"]["reset_frame_idx_each_step"] = False
params["env"]["env_build"]["reset_frame_idx"] = False

env = get_env(
  params['env_name'],
  params['env'])

if hasattr(env, "_obs_normalizer"):

  NORM_PATH = "{}/{}/{}/{}/model/_obs_normalizer_{}.pkl".format(
    args.log_dir,
    args.id,
    params['env_name'],
    args.seed,
    args.snap_check
  )
  with open(NORM_PATH, 'rb') as f:
    env._obs_normalizer = pickle.load(f)
    print(env._obs_normalizer._mean)
    print(env._obs_normalizer._var)


env.eval()

# params['net']['base_type'] = networks.MLPBase
params['net']['activation_func'] = torch.nn.ReLU
# params['net']['activation_func'] = torch.nn.ReLU

obs_normalizer = env._obs_normalizer if hasattr(env, "_obs_normalizer") \
  else None

encoder = networks.NatureFuseEncoder(
  in_channels=env.image_channels,
  state_input_dim=env.observation_space.shape[0],
  **params["encoder"]
)

pf = policies.GaussianContPolicyImpalaEncoderProj(
  encoder=encoder,
  state_input_shape=env.observation_space.shape[0],
  visual_input_shape=(env.image_channels, 64, 64),
  output_shape=env.action_space.shape[0],
  **params["net"],
  **params["policy"]
)

PATH = "{}/{}/{}/{}/model/model_pf_{}.pth".format(
  args.log_dir,
  args.id,
  params['env_name'],
  args.seed,
  args.snap_check
)

# PATH="log/MT10_MODULAR_4_4_2_NO_BN_GATED_REWEIGHT_RAND_CAS_PRESOFTMAX_ACTI_128/mt10/3/model/model_pf_best.pth"
# Value_PATH="log/MT10_MODULAR_4_4_2_NO_BN_GATED_REWEIGHT_RAND_CAS_PRESOFTMAX_ACTI_128/mt10/3/model/model_qf1_best.pth"
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

pf.eval()
# pf.load_state_dict(torch.load(PATH, map_location="cuda:0"))
# qf1.load_state_dict(torch.load(Value_PATH, map_location="cuda:0"))

action_weights = []

num_episodes = args.num_episodes

success = 0
count = 0
total_success = 0

# current_path = os.path.dirname(__file__)
# xml_root_path = os.path.join(current_path,"../torchmetal/envs/updated")
# from torchmetarl.envs.mujoco_dict import SETTING_DICT, ENV_DICT, TIMELIMIT_DICT
# def make_env(env_name, env_setting, env_build_params):
#     file_path = os.path.join(
#         os.path.join(xml_root_path, env_name),
#         SETTING_DICT[env_setting])
#     env = ENV_DICT[env_name](file_path, **env_build_params)
#     return env
# xml_root_path = "/"
# os.path.join(current_path,"updated")
# MODEL_CONDITION = [
#     [0, 0],
#     [0, 1],
#     [0, 2],
#     [1, 0],
#     [1, 1],
#     [1, 2],
#     [2, 0],
#     [2, 1],
#     [2, 2],
# ]

# REWARD_COEFF = [[np.cos(np.pi / 14 * i), np.sin(np.pi / 14 * i)] for i in range(8)]


rewards = []
task_names = []

video_output_path = "{}/{}/{}/{}".format(
  args.video_dir,
  args.id,
  params['env_name'],
  args.seed
)

Path(video_output_path).mkdir(parents=True, exist_ok=True)

# for idx in range(1):
for _ in range(1):
  # last_weights = []
  # general_weights = []
  # cat_weights = []
  # record_id = env.pybullet_client.startStateLogging(
  #     pybullet.STATE_LOGGING_VIDEO_MP4,
  #     "./record.mp4"
  # )
  import time
  t = time.time()
  count = 0
  morpho_action_weights = []

  env.seed(args.env_seed)
  random.seed(args.env_seed)
  torch.manual_seed(args.env_seed)
  np.random.seed(args.env_seed)
  # import pybullet as p
  # log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
  # env = wrappers.Monitor(env_single, "view_log", video_callable=None, force=True)

  from vidgear.gears import WriteGear
  output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}
  writer = WriteGear(
    output_filename=os.path.join(
      video_output_path, 'Output_{}{}.mp4'.format(args.snap_check, args.add_tag)),
    logging=True, **output_params
  )
  # out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20.00, (360, 480))
  # obs = env_single.reset()
  reward = 0
  # x_sped = 0
  # y_sped = 0
  step = 0
  for i in range(num_episodes):
    # env = get_env(
    #     "LaikagoMoveForward",
    #     "default",
    #     {
    #         "reward_scale": 1,
    #         "obs_norm": False,
    #         "env_build": {
    #             "enable_rendering_gui": True
    #         }
    #     }
    # )
    obs = env.reset()
    # env.pybullet_client.configureDebugVisualizer(
    #   env.pybullet_client.COV_ENABLE_RENDERING, 1)
    # success = 0
    while True:
      # env_single.render()
      frame_1 = obs[-64 * 64:]
      frame_1 = frame_1.reshape((64, 64))

      # import matplotlib.pyplot as plt
      # plt.imshow(frame_1, cmap='gray')
      # plt.pause(0.01)
      ob_t = torch.Tensor(obs).unsqueeze(0)
      # print(ob_t)
      # ob_t[0, -4:] =
      # ob_t[0, -4:] = 1
      # ob_t[0, -3] = 0
      # ob_t[0, -1] = 0
      # embedding_input = np.zeros(env_info.num_tasks)
      # embedding_input = torch.zeros(env_to_wrap.num_tasks)
      # embedding_input[idx] = 1
      # embedding_input = torch.cat([torch.Tensor(env_single.goal.copy()), embedding_input])
      # embedding_input = embedding_input.unsqueeze(0)
      # print(ob_t)
      # action, general_weight, last_weight = pf.eval_act(ob_t, embedding_input, return_weights = True )
      action = pf.eval_act(ob_t)
      # action = pf.explore(ob_t)
      # action = action["action"].squeeze(0).cpu().numpy()
      # print(action.shape)
      # print(action)
      count += 1
      # action = np.array([0,0,0,0,0,0])
      # action = -action
      # action = np.array([1,1,1,1,1,1])
      # print(action)
      #  = weights
      # last_weight = F.softmax(last_weight, dim=-1)
      # # last_weight = last_weight.exp()
      # general_weight = F.softmax(general_weight, dim = -1)
      # print(general_weight[0].shape)

      morpho_action_weights.append(action)
      import time
      # time.sleep(0.3)

      obs, rew, done, info = env.step(action)
      img = env.render(mode='rgb_array')

      sim_model = env.robot.quadruped
      pyb = env.pybullet_client
      root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)

      # import cv2
      # img = cv2.putText(
      #     img=np.copy(img),
      #     text="Speed X: {:.4f}".format(root_vel_sim[0]),
      #     org=(10, 30),
      #     fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

      # import cv2
      # img = cv2.putText(
      #     img=np.copy(img),
      #     text="Speed Y: {:.4f}".format(root_vel_sim[1]),
      #     org=(10, 50),
      #     fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

      # import cv2
      # img = cv2.putText(
      #     img=np.copy(img),
      #     text="Speed Z: {:.4f}".format(root_vel_sim[2]),
      #     org=(10, 70),
      #     fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

      # rot_quat = env.robot.GetBaseOrientation()
      # import cv2
      # img = cv2.putText(
      #     img=np.copy(img),
      #     text="Rot Quat 1: {:.4f}".format(rot_quat[0]),
      #     org=(200, 30),
      #     fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

      # import cv2
      # img = cv2.putText(
      #     img=np.copy(img),
      #     text="Rot Quat 1: {:.4f}".format(rot_quat[1]),
      #     org=(200, 50),
      #     fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

      # import cv2
      # img = cv2.putText(
      #     img=np.copy(img),
      #     text="Rot Quat 2: {:.4f}".format(rot_quat[2]),
      #     org=(200, 70),
      #     fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

      # import cv2
      # img = cv2.putText(
      #     img=np.copy(img),
      #     text="Rot Quat 3: {:.4f}".format(rot_quat[3]),
      #     org=(200, 90),
      #     fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)
      # plt.text(
      #     x=10,
      #     y=10,
      #     s="Speed: {}".format(root_vel_sim)
      # )
      # print(img.shape)
      # import matplotlib.pyplot as plt
      # plt.imshow(img)
      # plt.pause(0.01)
      # out.write(img)
      # print(img.shape)
      w, h, _ = img.shape
      frame_1 = (frame_1 - np.min(frame_1)) / \
        (np.max(frame_1) - np.min(frame_1))
      frame_resize = cv2.resize(frame_1, (w, w))
      frame_resize = (frame_resize * 255).astype(img.dtype)
      frame_resize = frame_resize.reshape((w, w, 1))
      frame_resize = np.repeat(frame_resize, 3, axis=2)
      # print(frame_resize)
      # import matplotlib.pyplot as plt
      # # plt.imshow(frame_resize)
      # # plt.imshow(img)
      # plt.imshow(np.concatenate([img, frame_resize], axis=1))
      # plt.pause(0.01)

      # print(np.vstack([img, frame_resize]).shape)
      # writer.write(frame_resize, rgb_mode=True)
      writer.write(np.concatenate(
        [img, frame_resize], axis=1), rgb_mode=True)
      # print(info["success"])
      # success = max(success, info["success"])
      # env.render()
      # if info["success"] and not done:
      # print("FXXk")
      # exit()
      reward += rew
      step += 1
      # x_sped += info["x_velocity"]
      # y_sped += info["y_velocity"]
      if done:

        # total_success += success
        # count += 1
        # print("EPoch:", idx, task_name, success, total_success / count)
        break
        # if not info["success"]:
        #     print("CAo")
        #     exit()
        # obs = env.reset()
        # idx += 1
    #         obs = env_single.reset()
    #         success = 0
    #         t = []
    #         t_now = 0
    # # m = [sin(t_now)]
    #         m = []
      # if len(info["rewards"]) > num_episodes:
      #     if len(info["rewards"]) == 1 and video_recorder.enabled:
      #         # save video of first episode
      #         print("Saved video.")
      #     print(info["rewards"][-1])
      #     num_episodes = le
    print("finish eposide")
  fps = count / (time.time() - t)
  print("Reward:", reward / num_episodes, "FPS:", fps)
  print("Step Counts:", step)

  # out.release()
  writer.close()
  # print("X Speed:", REWARD_COEFF[idx], x_sped / num_episodes)
  # print("Y Speed:", REWARD_COEFF[idx], y_sped / num_episodes)
  rewards.append(reward)
  # env.pybullet_client.stopStateLogging(log_id)
  # p.stopStateLogging(log_id)

  exit()
