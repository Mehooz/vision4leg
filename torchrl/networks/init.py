import numpy as np
import torch.nn as nn


def _fanin_init(tensor, alpha=0):
  size = tensor.size()
  if len(size) == 2:
    fan_in = size[0]
  elif len(size) > 2:
    fan_in = np.prod(size[1:])
  else:
    raise Exception("Shape must be have dimension at least 2.")
  # bound = 1. / np.sqrt(fan_in)
  bound = np.sqrt(1. / ((1 + alpha * alpha) * fan_in))
  return tensor.data.uniform_(-bound, bound)


def _uniform_init(tensor, param=3e-3):
  return tensor.data.uniform_(-param, param)


def _constant_bias_init(tensor, constant=0.1):
  tensor.data.fill_(constant)


def layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init):
  weight_init(layer.weight)
  bias_init(layer.bias)


def basic_init(layer):
  layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init)


def uniform_init(layer):
  layer_init(layer, weight_init=_uniform_init, bias_init=_uniform_init)


def _orthogonal_init(tensor, gain=np.sqrt(2)):
  nn.init.orthogonal_(tensor, gain=gain)


def orthogonal_init(layer, scale=np.sqrt(2), constant=0):
  layer_init(
    layer,
    weight_init=lambda x: _orthogonal_init(x, gain=scale),
    bias_init=lambda x: _constant_bias_init(x, 0))
