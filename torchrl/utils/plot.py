import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import sys
import os
from collections import OrderedDict
import tensorflow as tf
import argparse

from tensorflow.python.summary.summary_iterator import summary_iterator


def get_args():
  parser = argparse.ArgumentParser(description='RL')

  parser.add_argument('--seed', type=int, nargs='+', default=(0,),
                      help='random seed (default: (0,))'
                      )

  parser.add_argument('--env_name', type=str, default='HalfCheetah-v2',
                      help='environment to train on (default: HalfCheetah-v2)')

  parser.add_argument('--log_dir', type=str, default='./log',
                      help='directory for tensorboard logs (default: ./log)')

  parser.add_argument("--id", type=str, nargs='+', default=('origin',),
                      help="id for tensorboard")

  parser.add_argument('--output_dir', type=str, default='./fig',
                      help='directory for plot output (default: ./fig)')

  parser.add_argument('--entry', type=str, default='Running_Average_Rewards',
                      help='Record Entry')

  parser.add_argument('--add_tag', type=str, default='',
                      help='added tag')

  args = parser.parse_args()

  return args


args = get_args()
env_name = args.env_name
env_id = args.id


def get_name(path):
  print(path)
  return os.path.join(path, os.listdir(path)[0])


def post_process(array):
  smoth_para = 10
  new_array = []
  for i in range(len(array)):
    if i < len(array) - smoth_para:
      new_array.append(np.mean(array[i:i+smoth_para]))
    else:
      new_array.append(np.mean(array[i:None]))
  return new_array


plt.figure()
plt.figure(figsize=(10, 7))
colors = ['green', 'blue', 'red', 'orange', 'brown', 'purple', 'pink']
linestyles_choose = ['solid', 'solid', 'solid',
                     'solid', 'solid', 'solid', 'solid']

for eachcolor, eachlinestyle, exp_name in zip(colors, linestyles_choose, args.id):
  min_step_number = 1000000000000
  step_number = []
  all_scores = {}

  for seed in args.seed:
    file_path = os.path.join(env_name, str(seed))

    all_scores[seed] = []
    temp_step_number = []
    for e in summary_iterator(get_name(os.path.join(args.log_dir, exp_name, file_path))):
      for v in e.summary.value:
        if v.tag == args.entry:
          all_scores[seed].append(v.simple_value)
          temp_step_number.append(e.step)

    if temp_step_number[-1] < min_step_number:
      min_step_number = temp_step_number[-1]
      step_number = temp_step_number

  all_mean = []
  all_upper = []
  all_lower = []

  step_number = np.array(step_number) / 1e6
  final_step = []
  for i in range(len(step_number)):
    if step_number[i] <= 30:
      final_step.append(step_number[i])
      temp_list = []
      for key, valueList in all_scores.items():
        try:
          temp_list.append(valueList[i])
        except Exception:
          print(i)
          # exit()
      all_mean.append(np.mean(temp_list))
      all_upper.append(np.mean(temp_list) + np.std(temp_list))
      all_lower.append(np.mean(temp_list) - np.std(temp_list))
  all_mean = post_process(all_mean)
  all_lower = post_process(all_lower)
  all_upper = post_process(all_upper)
  # this_label = legend_base
  plt.plot(final_step, all_mean, label=exp_name,
           color=eachcolor, linestyle=eachlinestyle, linewidth=1)
  plt.plot(final_step, all_upper, color=eachcolor,
           linestyle=eachlinestyle, alpha=0.23, linewidth=0.5)
  plt.plot(final_step, all_lower, color=eachcolor,
           linestyle=eachlinestyle, alpha=0.23, linewidth=0.5)
  plt.fill_between(final_step, all_lower, all_upper,
                   facecolor=eachcolor, alpha=0.2)

plt.xlabel('Million Frames', fontsize=20)
plt.ylabel('Average for 10 episodes', fontsize=20)
plt.legend(loc='lower right', prop={'size': 12})
plt.title("{} {}".format(env_name, args.entry), fontsize=20)
if not os.path.exists(args.output_dir):
  os.mkdir(args.output_dir)
plt.savefig(os.path.join(args.output_dir, '{}_{}{}.png'.format(
  env_name, args.entry, args.add_tag)))
# plt.show()
plt.close()
