import os
import sys
import os.path as osp
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vision4leg.get_env import get_subprocvec_env
import random
import gym
from torchrl.collector.on_policy import VecOnPolicyCollector
from torchrl.algo import PPO, VMPO
import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import Logger
from torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from torchrl.utils import get_params
from torchrl.utils import get_args
import torch


args = get_args()
params = get_params(args.config)


def experiment(args):

  device = torch.device(
    "cuda:{}".format(args.device) if args.cuda else "cpu")

  env = get_subprocvec_env(
    params["env_name"],
    params["env"],
    args.vec_env_nums,
    args.proc_nums
  )
  eval_env = get_subprocvec_env(
    params["env_name"],
    params["env"],
    max(2, args.vec_env_nums),
    max(2, args.proc_nums),
  )

  if hasattr(env, "_obs_normalizer"):
    eval_env._obs_normalizer = env._obs_normalizer

  env.seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  buffer_param = params['replay_buffer']

  experiment_name = os.path.split(
    os.path.splitext(args.config)[0])[-1] if args.id is None \
    else args.id
  logger = Logger(
    experiment_name, params['env_name'],
    args.seed, params, args.log_dir, args.overwrite)
  params['general_setting']['env'] = env

  replay_buffer = OnPolicyReplayBuffer(
    env_nums=args.vec_env_nums,
    max_replay_buffer_size=int(buffer_param['size']),
    time_limit_filter=buffer_param['time_limit_filter']
  )
  params['general_setting']['replay_buffer'] = replay_buffer

  params['general_setting']['logger'] = logger
  params['general_setting']['device'] = device

  params['net']['base_type'] = networks.MLPBase
  # params['net']['activation_func'] = torch.nn.Tanh

  encoder = networks.TransformerEncoder(
    in_channels=env.image_channels,
    **params["encoder"]
  )

  pf = policies.GaussianContPolicyTransformer(
    encoder=encoder,
    visual_input_shape=(env.image_channels, 64, 64),
    output_shape=env.action_space.shape[0],
    **params["net"],
    **params["policy"]
  )

  vf = networks.Transformer(
    encoder=encoder,
    visual_input_shape=(env.image_channels, 64, 64),
    output_shape=1,
    **params["net"]
  )

  print(pf)
  print(vf)

  params['general_setting']['collector'] = VecOnPolicyCollector(
    vf, env=env, eval_env=eval_env, pf=pf,
    replay_buffer=replay_buffer, device=device,
    train_render=False,
    **params["collector"]
  )
  params['general_setting']['save_dir'] = osp.join(
    logger.work_dir, "model")
  agent = PPO(
    pf=pf,
    vf=vf,
    **params["ppo"],
    **params["general_setting"]
  )
  agent.train()


if __name__ == "__main__":
  experiment(args)
