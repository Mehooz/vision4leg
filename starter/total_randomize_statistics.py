import sys
sys.path.append(".")
import vision4leg.robots.laikago_constants as lc
import pickle
import pybullet
import numpy as np
import time
import matplotlib.pyplot as plt
import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import get_params
from vision4leg.get_env import get_env
import seaborn as sns
import random
import argparse
import torch
import os


def get_args():
  parser = argparse.ArgumentParser(description='RL')
  parser.add_argument('--seed', type=int, nargs='+', default=(0,),
                      help='random seed (default: (0,))')
  parser.add_argument('--env_seed', type=int, default=0,
                      help='random seed (default: 1)')
  parser.add_argument('--env_name', type=str,
                      default='A1MoveGround')
  parser.add_argument('--num_episodes', type=int, default=3,
                      help='number of episodes')
  parser.add_argument("--config", type=str, default=None,
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
  parser.add_argument("--interval", type=int, default=100)
  parser.add_argument("--max_model", type=int, default=2000)
  parser.add_argument('--snap_check', type=str, default='best')

  # tensorboard
  parser.add_argument("--id", type=str, default=None,
                      help="id for tensorboard", )

  args = parser.parse_args()

  args.cuda = not args.no_cuda and torch.cuda.is_available()

  return args


args = get_args()


np.random.seed(0)
random.seed(0)

PARAM_PATH = "{}/{}/{}/{}/params.json".format(
  args.log_dir,
  args.id,
  args.env_name,
  args.seed[0]
)
# env_params = get_params(args.config)

params = get_params(PARAM_PATH)
env_params = params

env_params["env"]["env_build"]["reset_frame_idx_each_step"] = True
if "get_image_interval" in env_params["env"]["env_build"] and env_params["env"]["env_build"]["get_image_interval"] > 1:
  env_params["env"]["env_build"]["frame_extract"] = env_params["env"]["env_build"]["get_image_interval"]
  env_params["env"]["env_build"]["get_image_interval"] = 1
elif "get_image_interval" in env_params["env"]["env_build"] and env_params["env"]["env_build"]["get_image_interval"] == 1 and env_params["env"]["env_build"]["frame_extract"] == 1:
  env_params["env"]["env_build"]["frame_extract"] = 4

if "curriculum" in env_params["env"]["env_build"]:
  env_params["env"]["env_build"]["curriculum"] = False
env = get_env(
  env_params['env_name'],
  env_params['env']
)

if hasattr(env, "_obs_normalizer"):
  NORM_PATH = "{}/{}/{}/{}/model/_obs_normalizer_{}.pkl".format(
    args.log_dir,
    args.id,
    params['env_name'],
    args.seed[0],
    args.snap_check
  )
  with open(NORM_PATH, 'rb') as f:
    env._obs_normalizer = pickle.load(f)

env.eval()


# params['net']['base_type'] = networks.MLPBase
params['net']['activation_func'] = torch.nn.ReLU
# params['net']['activation_func'] = torch.nn.ReLU

obs_normalizer = env._obs_normalizer if hasattr(env, "_obs_normalizer") \
  else None

# encoder = networks.NatureHRLEncoder(
#     in_channels=env.image_channels,
#     state_input_dim=env.observation_space.shape[0],
#     **params["encoder"]
# )

# pf = policies.GaussianContPolicyHRL(
#     encoder=encoder,
#     state_input_shape=env.observation_space.shape[0],
#     visual_input_shape=(env.image_channels, 64, 64),
#     output_shape=env.action_space.shape[0],
#     **params["net"],
#     **params["policy"]
# )
params['net']['activation_func'] = torch.nn.ReLU

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

test_results = []
goal_collect_count = []
collision_count_result = []

distance_result = []

model_check = args.snap_check

# for model_check in range(0, args.max_model, args.interval):
reward_result = []
goal_result = []
collision_result = []
distance_result = []

for seed in args.seed:
  if hasattr(env, "_obs_normalizer"):
    NORM_PATH = "{}/{}/{}/{}/model/_obs_normalizer_{}.pkl".format(
      args.log_dir,
      args.id,
      params['env_name'],
      seed,
      model_check
    )
    with open(NORM_PATH, 'rb') as f:
      env._obs_normalizer = pickle.load(f)
  env.eval()

  PATH = "{}/{}/{}/{}/model/model_pf_{}.pth".format(
    args.log_dir,
    args.id,
    params['env_name'],
    seed,
    model_check
  )

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

  num_episodes = args.num_episodes

  count = 0

  import pybullet
  import cv2

  import time
  t = time.time()
  count = 0

  env.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

  reward = 0
  goal_count = 0
  collision_count = 0
  distance = 0
  step = 0
  for i in range(num_episodes):
    obs = env.reset()
    sim_model = env.robot.quadruped
    pyb = env.pybullet_client
    root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)

    start_pos, _ = pyb.getBasePositionAndOrientation(sim_model)

    while True:
      ob_t = torch.Tensor(obs).unsqueeze(0)

      action = pf.eval_act(ob_t)

      count += 1

      obs, rew, done, info = env.step(action)

      if rew > 90:
        goal_count += 1

      sim_model = env.robot.quadruped
      pyb = env.pybullet_client
      root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)

      # collision = 0
      contacts = env.pybullet_client.getContactPoints(
        bodyA=env.robot.quadruped)
      # for contact in contacts:
      for contact in contacts:
        if contact[2] is not env.world_dict["ground"]:
          collision_count += 1
          break
        if contact[2] is env.world_dict["ground"]:
          if contact[3] not in env.robot._foot_link_ids:
            collision_count += 1
            # print()
            break
      # print(collision_count)

      rot_quat = env.robot.GetBaseOrientation()

      reward += rew
      step += 1

      if done:
        break

    end_pos, _ = pyb.getBasePositionAndOrientation(sim_model)
    distance += end_pos[0] - start_pos[0]

  fps = count / (time.time() - t)
  print("Seed: {:3}, {:6} Reward: {:8.2f}, Steps: {:4},  FPS: {:6f}, Goal: {:4}, Collision: {: 4}, Distance: {:8.2f}".format(
    seed, model_check, reward / num_episodes, step, fps, goal_count /
    num_episodes, collision_count / num_episodes, distance / num_episodes
  ))
  reward_result.append(reward / num_episodes)
  goal_result.append(goal_count / num_episodes * 100)
  collision_result.append(collision_count / num_episodes)
  distance_result.append(distance / num_episodes)

# test_results.append(per_model_result)
# goal_collect_count.append(goal_per_model_result)
# collision_count_result.append(collision_per_model_result)
# distance_result.append(distance_per_model_result)
# print("Reward Results")

# print(test_results)

# print("Goal Collected")

# print(goal_collect_count)

# print("Collision Happened")
# print(collision_count_result)

# print("Distance Moved")
# print(distance_result)

output_dic = {
  "Reward_results": {
    "mean": np.mean(reward_result),
    "std": np.std(reward_result),
  },
  "goal_results": {
    "mean": np.mean(goal_result),
    "std": np.std(goal_result),
  },
  "collision_results": {
    "mean": np.mean(collision_result),
    "std": np.std(collision_result),
  },
  "distance_results": {
    "mean": np.mean(distance_result),
    "std": np.std(distance_result),
  },
}

print(output_dic)
with open("./stats/{}.pkl".format(args.id), "wb") as f:
  pickle.dump(output_dic, f)
