import argparse
import json
import torch


def get_args():
  parser = argparse.ArgumentParser(description='RL')

  parser.add_argument('--seed', type=int, default=0,
                      help='random seed (default: 1)')

  parser.add_argument('--vec_env_nums', type=int, default=4,
                      help='vec env nums')

  parser.add_argument('--proc_nums', type=int, default=4,
                      help='vec env nums')

  parser.add_argument('--eval_worker_nums', type=int, default=2,
                      help='eval worker nums')

  parser.add_argument("--config", type=str,   default=None,
                      help="config file",)

  parser.add_argument('--save_dir', type=str, default='./snapshots',
                      help='directory for snapshots (default: ./snapshots)')

  parser.add_argument('--log_dir', type=str, default='./log',
                      help='directory for tensorboard logs (default: ./log)')

  parser.add_argument('--no_cuda', action='store_true', default=False,
                      help='disables CUDA training')

  parser.add_argument('--overwrite', action='store_true', default=False,
                      help='overwrite previous experiments')

  parser.add_argument("--device", type=int, default=0,
                      help="gpu secification",)

  # tensorboard
  parser.add_argument("--id", type=str,   default=None,
                      help="id for tensorboard",)

  args = parser.parse_args()

  args.cuda = not args.no_cuda and torch.cuda.is_available()

  return args


def get_params(file_name):
  with open(file_name) as f:
    params = json.load(f)
  return params
