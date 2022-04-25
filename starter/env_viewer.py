import sys
sys.path.append(".")
import torch
import argparse
from vision4leg.get_env import get_env
from torchrl.utils import get_params
import gym
import gym.wrappers as wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time


def get_args():
  parser = argparse.ArgumentParser(description='RL')

  parser.add_argument('--seed', type=int, default=0,
                      help='random seed (default: 1)')

  parser.add_argument('--env_seed', type=int, default=0,
                      help='random seed (default: 1)')

  parser.add_argument('--num_episodes', type=int, default=1,
                      help='number of episodes')

  parser.add_argument("--config", type=str, default=None,
                      help="config file", )

  parser.add_argument('--save_dir', type=str, default='./snapshots',
                      help='directory for snapshots (default: ./snapshots)')

  parser.add_argument('--log_dir', type=str, default='./log',
                      help='directory for tensorboard logs (default: ./log)')

  parser.add_argument('--no_cuda', action='store_true', default=False,
                      help='disables CUDA training')

  parser.add_argument("--device", type=int, default=0,
                      help="gpu secification", )

  parser.add_argument('--add_tag', type=str, default='_mlp',
                      help='directory for snapshots (default: ./snapshots)')

  # tensorboard
  parser.add_argument("--id", type=str, default=None,
                      help="id for tensorboard", )

  args = parser.parse_args()

  args.cuda = not args.no_cuda and torch.cuda.is_available()

  return args


args = get_args()

params = get_params(args.config)

start_time = time.time()

params["env"]["env_build"]["enable_rendering"] = True
# params["env"]["env_build"]["moving"] = True
env = get_env(params['env_name'], params['env'])

action_weights = []

num_episodes = args.num_episodes

success = 0
count = 0
total_success = 0

rewards = []
task_names = []
for i in range(num_episodes):
  obs = env.reset()
  reward = 0
  for i in range(500):
    obs, rew, done, info = env.step(env.action_space.sample())
    reward += rew
    if done:
      break
  print("finish eposide")

rewards.append(reward)
end_time = time.time()
print("total time is {}".format(end_time - start_time))
