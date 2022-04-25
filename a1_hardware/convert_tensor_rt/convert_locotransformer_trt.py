import torch
import torchrl.networks.nets as networks
import torchrl.networks.base as networks_base
import torchrl.policies as policies
from torchrl.utils import get_params
# from rich4loco.get_env import get_env

import glob
import os

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--save_dir', type=str, default='./onnx')
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard",)
    args = parser.parse_args()
    return args

args = get_args()

SEED = args.seed
POLICYNAME = args.id

data_save_dir = "onnx/"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)

save_name = os.path.join(
    data_save_dir, "{}_seed{}_half.onnx".format(POLICYNAME, SEED)
)

PARAM_PATH = args.log_dir + "/policies/" + POLICYNAME + "/A1MoveGround/" + str(SEED) + "/params.json"
NORM_PATH = args.log_dir + "/policies/" + POLICYNAME + "/A1MoveGround/" + str(SEED) + "/model/_obs_normalizer_best.pkl"

params = get_params(PARAM_PATH)
params['net']['activation_func'] = torch.nn.ReLU
use_foot_contact = params['env']['env_build']['add_foot_contact']

params['net']['base_type'] = networks_base.MLPBase

encoder = networks_base.LocoTransformerEncoder(
    in_channels=4,
    state_input_dim=84 if not use_foot_contact else 96,
    **params["encoder"]
)

class LocoTransPolicyExecutor(policies.GaussianContPolicyLocoTransformer):
    def forward(self, x):
        mean = networks.LocoTransformer.forward(self, x)
        return mean

pf = LocoTransPolicyExecutor(
    encoder=encoder,
    state_input_shape=84 if not use_foot_contact else 96,
    visual_input_shape=(4, 64, 64),
    output_shape=6,
    **params["net"],
    **params["policy"]
)

PATH = args.log_dir + "/policies/" + POLICYNAME + "/A1MoveGround/" + str(SEED) + "/model/model_pf_best.pth"

pf.load_state_dict( 
    torch.load(
        PATH,
        map_location="cuda:0"
    )
)
pf.to("cuda:0")

pf = pf.half()
print()
print(pf(torch.rand(1, (84 if not use_foot_contact else 96) + 64 * 64 * 4).to("cuda:0").half()))


import torch
import numpy as np

BATCH_SIZE = 1
state_dim = 84 if not use_foot_contact else 96
visual_dim = (4, 64, 64)
dummy_input = torch.randn(1,  np.prod(visual_dim) + state_dim).to("cuda:0").half()

import torch.onnx
torch.onnx.export(pf, dummy_input, save_name, verbose=False)