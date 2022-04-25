import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchrl.networks.init as init


class MLPBase(nn.Module):
  def __init__(
      self,
      input_shape,
      hidden_shapes,
      activation_func=nn.ReLU,
      init_func=init.basic_init,
      add_ln=False,
      last_activation_func=None):
    super().__init__()

    self.activation_func = activation_func
    self.fcs = []
    self.add_ln = add_ln
    if last_activation_func is not None:
      self.last_activation_func = last_activation_func
    else:
      self.last_activation_func = activation_func
    input_shape = np.prod(input_shape)

    self.output_shape = input_shape
    for next_shape in hidden_shapes:
      fc = nn.Linear(input_shape, next_shape)
      init_func(fc)
      self.fcs.append(fc)
      self.fcs.append(activation_func())
      if self.add_ln:
        self.fcs.append(nn.LayerNorm(next_shape))
      input_shape = next_shape
      self.output_shape = next_shape

    self.fcs.pop(-1)
    self.fcs.append(self.last_activation_func())
    self.seq_fcs = nn.Sequential(*self.fcs)

  def forward(self, x):
    return self.seq_fcs(x)


def calc_next_shape(input_shape, conv_info):
  """
  take input shape per-layer conv-info as input
  """
  out_channels, kernel_size, stride, padding = conv_info
  _, h, w = input_shape
  # for padding, dilation, kernel_size, stride in conv_info:
  h = int((h + 2*padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1)
  w = int((w + 2*padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)
  return (out_channels, h, w)


class CNNBase(nn.Module):
  def __init__(
      self, input_shape,
      hidden_shapes,
      activation_func=nn.ReLU,
      init_func=init.basic_init,
      add_ln=False,
      last_activation_func=None):
    super().__init__()

    current_shape = input_shape
    in_channels = input_shape[0]
    self.add_ln = add_ln
    self.activation_func = activation_func
    if last_activation_func is not None:
      self.last_activation_func = last_activation_func
    else:
      self.last_activation_func = activation_func
    self.convs = []
    self.output_shape = current_shape[0] * \
      current_shape[1] * current_shape[2]

    for conv_info in hidden_shapes:
      out_channels, kernel_size, stride, padding = conv_info
      conv = nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding)
      init_func(conv)

      self.convs.append(conv)
      self.convs.append(activation_func())
      in_channels = out_channels
      current_shape = calc_next_shape(current_shape, conv_info)
      if self.add_ln:
        self.convs.append(nn.LayerNorm(current_shape[1:]))

      self.output_shape = current_shape[0] * \
        current_shape[1] * current_shape[2]

    self.convs.pop(-1)
    self.convs.append(self.last_activation_func())
    self.seq_convs = nn.Sequential(*self.convs)

  def forward(self, x):
    view_shape = x.size()[:-3] + torch.Size([-1])
    x = x.view(torch.Size(
      [np.prod(x.size()[:-3])]) + x.size()[-3:])
    out = self.seq_convs(x)
    return out.view(view_shape)


# From https://github.com/joonleesky/train-procgen-pytorch/blob/1678e4a9e2cb8ffc3772ecb3b589a3e0e06a2281/common/model.py#L94

def xavier_uniform_init(module, gain=1.0):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.xavier_uniform_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module


class ResidualBlock(nn.Module):
  def __init__(self,
               in_channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(
      in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(
      in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    out = nn.ReLU()(x)
    out = self.conv1(out)
    out = nn.ReLU()(out)
    out = self.conv2(out)
    return out + x


class ImpalaBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ImpalaBlock, self).__init__()
    self.conv = nn.Conv2d(
      in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
    self.res1 = ResidualBlock(out_channels)
    self.res2 = ResidualBlock(out_channels)

  def forward(self, x):
    x = self.conv(x)
    x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
    x = self.res1(x)
    x = self.res2(x)
    return x


class Flatten(nn.Module):
  def forward(self, x):
    # base_shape = v.shape[-2]
    return x.view(x.size(0), -1)


class ImpalaEncoder(nn.Module):
  def __init__(self,
               in_channels,
               flatten=True,
               **kwargs):
    super(ImpalaEncoder, self).__init__()
    self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
    self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
    self.block3 = ImpalaBlock(in_channels=32, out_channels=32)

    self.flatten = flatten

    self.output_dim = 32 * 8 * 8
    self.apply(xavier_uniform_init)

  def forward(self, x, detach=False):

    view_shape = x.size()[:-3] + torch.Size([-1])
    x = x.view(torch.Size(
      [np.prod(x.size()[:-3])]) + x.size()[-3:])
    # out = self.seq_convs(x)
    # return out

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = nn.ReLU()(x)
    if self.flatten:
      x = Flatten()(x).view(view_shape)
    if detach:
      x = x.detach()
    return x


def weight_init(m):
  """Custom weight init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)
  elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class RLProjection(nn.Module):
  def __init__(self, in_dim, out_dim, proj=True):
    super().__init__()
    self.out_dim = out_dim
    module_list = [
      nn.Linear(in_dim, out_dim)
    ]
    if proj:
      module_list += [
        # nn.LayerNorm(out_dim),
        # nn.Tanh()
        nn.ReLU()
      ]

    self.projection = nn.Sequential(
      *module_list
    )
    self.output_dim = out_dim
    self.apply(weight_init)

  def forward(self, x):
    return self.projection(x)


class RLPredictor(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim=512):
    super().__init__()
    self.out_dim = out_dim
    module_list = [
      nn.Linear(in_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, out_dim),
    ]

    self.predict = nn.Sequential(
      *module_list
    )
    self.output_dim = out_dim
    self.apply(weight_init)

  def forward(self, x):
    return self.predict(x)


class ImpalaFuseEncoder(nn.Module):
  def __init__(self,
               in_channels,
               state_input_dim,
               visual_dim,
               hidden_shapes,
               proj=True,
               **kwargs):
    super(ImpalaFuseEncoder, self).__init__()
    self.visual_base = ImpalaEncoder(
      in_channels
    )

    self.visual_dim = visual_dim
    self.visual_projector = RLProjection(
      in_dim=self.visual_base.output_dim,
      out_dim=visual_dim
    )

    self.base = MLPBase(
      input_shape=state_input_dim,
      hidden_shapes=hidden_shapes,
      # last_activation_func=nn.Tanh,
      **kwargs
    )

  def forward(self, visual_x, state_x, detach=False):
    if len(visual_x.shape) <= 3:
      visual_x = visual_x.unsqueeze(0)
      state_x = state_x.unsqueeze(0)

    view_shape = visual_x.size()[:-3] + torch.Size([-1])
    visual_x = visual_x.view(torch.Size(
      [np.prod(visual_x.size()[:-3])]) + visual_x.size()[-3:]
    )

    visual_out = self.visual_base(visual_x, detach=detach)
    visual_out = self.visual_projector(visual_out)

    state_out = self.base(state_x)

    return visual_out, state_out


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.orthogonal_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module


class NatureEncoder(nn.Module):
  def __init__(self,
               in_channels,
               groups=1,
               flatten=True,
               **kwargs):
    """
    input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
    filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
    use_batchnorm: (bool) whether to use batchnorm
    """
    super(NatureEncoder, self).__init__()
    self.groups = groups
    layer_list = [
      nn.Conv2d(in_channels=in_channels, out_channels=32 * self.groups,
                kernel_size=8, stride=4), nn.ReLU(),
      nn.Conv2d(in_channels=32 * self.groups, out_channels=64 * self.groups,
                kernel_size=4, stride=2), nn.ReLU(),
      nn.Conv2d(in_channels=64 * self.groups, out_channels=64 * self.groups,
                kernel_size=3, stride=1), nn.ReLU(),
    ]
    if flatten:
      layer_list.append(
        Flatten()
      )
    self.layers = nn.Sequential(*layer_list)

    self.output_dim = 1024 * self.groups
    self.apply(orthogonal_init)

  def forward(self, x, detach=False):
    # view_shape = x.size()[:-3] + torch.Size([-1])
    x = x.view(torch.Size(
      [np.prod(x.size()[:-3])]) + x.size()[-3:])
    x = self.layers(x)
    # x = x.view(view_shape)
    if detach:
      x = x.detach()
    return x


class NatureFuseEncoder(nn.Module):
  def __init__(self,
               in_channels,
               state_input_dim,
               visual_dim,
               hidden_shapes,
               proj=True,
               **kwargs):
    super(NatureFuseEncoder, self).__init__()
    self.visual_base = NatureEncoder(
      in_channels
    )

    self.visual_dim = visual_dim
    self.visual_projector = RLProjection(
      in_dim=self.visual_base.output_dim,
      out_dim=visual_dim
    )

    self.base = MLPBase(
      input_shape=state_input_dim,
      hidden_shapes=hidden_shapes,
      # last_activation_func=nn.Tanh,
      **kwargs
    )

  def forward(self, visual_x, state_x, detach=False):
    if len(visual_x.shape) <= 3:
      visual_x = visual_x.unsqueeze(0)
      state_x = state_x.unsqueeze(0)

    visual_x = visual_x.view(torch.Size(
      [np.prod(visual_x.size()[:-3])]) + visual_x.size()[-3:]
    )

    visual_out = self.visual_base(visual_x, detach=detach)
    visual_out = self.visual_projector(visual_out)

    state_out = self.base(state_x)

    return visual_out, state_out


class TransformerEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      token_dim=64,
      two_by_two=False,
      **kwargs
  ):
    super(TransformerEncoder, self).__init__()
    # assert in_channels == 16
    self.in_channels = in_channels
    self.token_dim = token_dim
    self.two_by_two = two_by_two
    if self.in_channels == 12 or self.in_channels == 16:
      self.rgb_visual_base = NatureEncoder(
        12, flatten=False
      )
      if two_by_two:
        self.rgb_up_conv = nn.Conv2d(64, token_dim, 2, stride=2)
      else:
        self.rgb_up_conv = nn.Conv2d(64, token_dim, 1)

    if self.in_channels == 4 or self.in_channels == 16:
      self.depth_visual_base = NatureEncoder(
        4, flatten=False
      )
      # if token_dim != 64:
      if two_by_two:
        self.depth_up_conv = nn.Conv2d(64, token_dim, 2, stride=2)
      else:
        self.depth_up_conv = nn.Conv2d(64, token_dim, 1)

    self.visual_dim = token_dim  # RGBD(DEPTH) + POSITIONAL
    self.per_modal_tokens = 16
    if self.two_by_two:
      self.per_modal_tokens = 4

    self.flatten_layer = Flatten()

  def forward(self, visual_x, detach=False, return_raw_visual_vecs=False):
    if len(visual_x.shape) <= 3:
      visual_x = visual_x.unsqueeze(0)

    visual_x = visual_x.view(torch.Size(
      [np.prod(visual_x.size()[:-3])]) + visual_x.size()[-3:]
    )
    # (Batch Shape, 16, 64, 64)

    if self.in_channels == 12 or self.in_channels == 16:
      rgb_visual_x = visual_x[..., :12, :, :]
    if self.in_channels == 16:
      depth_visual_x = visual_x[..., 12:, :, :]
    if self.in_channels == 4:
      depth_visual_x = visual_x[..., :4, :, :]

    raw_visual_vecs = []

    if self.in_channels == 12 or self.in_channels == 16:
      rgb_visual_out_raw = self.rgb_visual_base(
        rgb_visual_x, detach=detach)
      # if self.token_dim != 64:
      rgb_visual_out = self.rgb_up_conv(rgb_visual_out_raw)
      raw_visual_vecs.append(self.flatten_layer(rgb_visual_out_raw))

    if self.in_channels == 4 or self.in_channels == 16:
      depth_visual_out_raw = self.depth_visual_base(
        depth_visual_x, detach=detach)
      # if self.token_dim != 64:
      depth_visual_out = self.depth_up_conv(depth_visual_out_raw)
      raw_visual_vecs.append(self.flatten_layer(depth_visual_out_raw))

    # (Batch Shape, Channel, # Patches, # Patches)
    if self.in_channels == 12 or self.in_channels == 16:
      visual_shape = rgb_visual_out.shape
    else:
      visual_shape = depth_visual_out.shape
    # (Batch Shape, Channel, # Patches, # Patches)
    num_patches = visual_shape[-1]

    if self.in_channels == 12 or self.in_channels == 16:
      rgb_visual_out = rgb_visual_out.reshape(
        visual_shape[0], visual_shape[1], num_patches * num_patches
      )
      # (Batch Shape, Channel, # Patches ** 2)
      rgb_visual_out = rgb_visual_out.permute(
        2, 0, 1
      )
      # (# Patches ** 2， Batch Shape, Channel)
    if self.in_channels == 4 or self.in_channels == 16:
      depth_visual_out = depth_visual_out.reshape(
        visual_shape[0], visual_shape[1], num_patches * num_patches
      )
      # (Batch Shape, Channel, # Patches ** 2)
      depth_visual_out = depth_visual_out.permute(
        2, 0, 1
      )
      # (# Patches ** 2， Batch Shape, Channel)
    if self.in_channels == 4:
      visual_out = depth_visual_out
    elif self.in_channels == 12:
      visual_out = rgb_visual_out
    elif self.in_channels == 16:
      visual_out = torch.cat([depth_visual_out, rgb_visual_out], dim=0)

    if return_raw_visual_vecs:
      return visual_out, raw_visual_vecs
    return visual_out


class LocoTransformerEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      state_input_dim,
      hidden_shapes,
      token_dim=64,
      two_by_two=False,
      visual_dim=None,
      proj=True,
      **kwargs
  ):
    super(LocoTransformerEncoder, self).__init__()
    # assert in_channels == 16
    self.in_channels = in_channels
    self.token_dim = token_dim
    self.two_by_two = two_by_two
    if self.in_channels == 12 or self.in_channels == 16:
      self.rgb_visual_base = NatureEncoder(
        12, flatten=False
      )
      if two_by_two:
        self.rgb_up_conv = nn.Conv2d(64, token_dim, 2, stride=2)
      else:
        self.rgb_up_conv = nn.Conv2d(64, token_dim, 1)

    if self.in_channels == 4 or self.in_channels == 16:
      self.depth_visual_base = NatureEncoder(
        4, flatten=False
      )
      # if token_dim != 64:
      if two_by_two:
        self.depth_up_conv = nn.Conv2d(64, token_dim, 2, stride=2)
      else:
        self.depth_up_conv = nn.Conv2d(64, token_dim, 1)

    self.base = MLPBase(
      input_shape=state_input_dim,
      hidden_shapes=hidden_shapes,
      # last_activation_func=nn.Tanh,
      **kwargs
    )
    self.state_projector = RLProjection(
      in_dim=self.base.output_shape,
      out_dim=token_dim
    )
    self.visual_dim = token_dim  # RGBD(DEPTH) + POSITIONAL
    self.per_modal_tokens = 16
    if self.two_by_two:
      self.per_modal_tokens = 4

    self.flatten_layer = Flatten()

  def forward(self, visual_x, state_x, detach=False, return_raw_visual_vecs=False):
    if len(visual_x.shape) <= 3:
      visual_x = visual_x.unsqueeze(0)
      state_x = state_x.unsqueeze(0)

    view_shape = visual_x.size()[:-3] + torch.Size([-1])
    visual_x = visual_x.view(torch.Size(
      [np.prod(visual_x.size()[:-3])]) + visual_x.size()[-3:]
    )
    # (Batch Shape, 16, 64, 64)

    if self.in_channels == 12 or self.in_channels == 16:
      rgb_visual_x = visual_x[..., :12, :, :]
    if self.in_channels == 16:
      depth_visual_x = visual_x[..., 12:, :, :]
    if self.in_channels == 4:
      depth_visual_x = visual_x[..., :4, :, :]

    raw_visual_vecs = []

    if self.in_channels == 12 or self.in_channels == 16:
      rgb_visual_out_raw = self.rgb_visual_base(
        rgb_visual_x, detach=detach)
      # if self.token_dim != 64:
      rgb_visual_out = self.rgb_up_conv(rgb_visual_out_raw)
      raw_visual_vecs.append(self.flatten_layer(rgb_visual_out_raw))

    if self.in_channels == 4 or self.in_channels == 16:
      depth_visual_out_raw = self.depth_visual_base(
        depth_visual_x, detach=detach)
      # if self.token_dim != 64:
      depth_visual_out = self.depth_up_conv(depth_visual_out_raw)
      raw_visual_vecs.append(self.flatten_layer(depth_visual_out_raw))

    # (Batch Shape, Channel, # Patches, # Patches)
    if self.in_channels == 12 or self.in_channels == 16:
      visual_shape = rgb_visual_out.shape
    else:
      visual_shape = depth_visual_out.shape
    # (Batch Shape, Channel, # Patches, # Patches)
    num_patches = visual_shape[-1]

    if self.in_channels == 12 or self.in_channels == 16:
      rgb_visual_out = rgb_visual_out.reshape(
        visual_shape[0], visual_shape[1], num_patches * num_patches
      )
      # (Batch Shape, Channel, # Patches ** 2)
      rgb_visual_out = rgb_visual_out.permute(
        2, 0, 1
      )
      # (# Patches ** 2， Batch Shape, Channel)
    if self.in_channels == 4 or self.in_channels == 16:
      depth_visual_out = depth_visual_out.reshape(
        visual_shape[0], visual_shape[1], num_patches * num_patches
      )
      # (Batch Shape, Channel, # Patches ** 2)
      depth_visual_out = depth_visual_out.permute(
        2, 0, 1
      )
      # (# Patches ** 2， Batch Shape, Channel)

    state_out = self.base(state_x)

    state_out_proj = self.state_projector(state_out)
    # (Batch Shape, Channel)
    state_out_proj = state_out_proj.unsqueeze(0)

    out_list = [state_out_proj]
    if self.in_channels == 12 or self.in_channels == 16:
      out_list.append(rgb_visual_out)
    if self.in_channels == 4 or self.in_channels == 16:
      out_list.append(depth_visual_out)
    visual_out = torch.cat(out_list, dim=0)

    if return_raw_visual_vecs:
      return visual_out, state_out, raw_visual_vecs
    return visual_out, state_out
