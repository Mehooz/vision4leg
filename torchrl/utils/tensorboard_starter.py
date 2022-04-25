import time
import os
import sys
import subprocess
import argparse


def checkNotFinish(popen_list):
  for eachpopen in popen_list:
    if eachpopen.poll() == None:
      return True
  return False


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--port', type=int, default=6006,
                    help='port to run the server on (default: 8097)')
parser.add_argument("--id", type=str, nargs='+', default=('origin',),
                    help="id for tensorboard")
parser.add_argument('--seed', type=int, nargs='+', default=(0,),
                    help='random seed (default: (0,))')
parser.add_argument('--env_name', type=str, default="HalfCheetah-v2")
parser.add_argument('--base_log_dir', type=str, default="./log")
args = parser.parse_args()

base_command = "tensorboard --logdir="

for exp in args.id:
  for seed in args.seed:
    base_command += "{1}-{2}-{3}:{0}/{1}/{2}/{3},".format(
      args.base_log_dir, exp, args.env_name, seed)

base_command = base_command[:-1]
base_command = base_command+" --port {}".format(args.port)

p = subprocess.Popen(base_command, shell=True)

while True:
  if p.poll() == None:
    break
