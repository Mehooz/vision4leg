
from torchrl.env.continuous_wrapper import *
from torchrl.env.base_wrapper import *
from torchrl.env.vecenv import VecEnv
from torchrl.env.subproc_vecenv import SubProcVecEnv
from gym.wrappers.time_limit import TimeLimit
from vision4leg.env_dict import ENV_DICT, TIMELIMIT_DICT


def wrap_continuous_env(env, obs_norm, reward_scale):
  env = RewardShift(env, reward_scale)
  if obs_norm:
    return NormObs(env)
  return env


class ActClip(gym.ActionWrapper, BaseWrapper):
  """
  Normalized Action      => [ -1, 1 ]
  """

  def __init__(self, env, clip_num=0.3):
    super(ActClip, self).__init__(env)
    # ub = np.ones(self.env.action_space.shape)
    # self.action_space = gym.spaces.Box(-1 * ub, ub)
    # self.lb = self.env.action
    # n_space.low
    # self.ub = self.env.action_space.high
    self.clip_num = clip_num

  def action(self, action):
    # scaled_action = self.lb + (action + 1.) * 0.5 * (self.ub - self.lb)
    return np.clip(action, -self.clip_num, self.clip_num)


def make_env(env_name, env_build_params):
  env = ENV_DICT[env_name](**env_build_params)
  return env


class NormObsWithImg(gym.ObservationWrapper, BaseWrapper):
  """
  Normalized Observation => Optional, Use Momentum
  """

  def __init__(self, env, epsilon=1e-4, clipob=10.):
    super(NormObsWithImg, self).__init__(env)
    self.count = epsilon
    self.clipob = clipob
    self._obs_normalizer = Normalizer(env.observation_space.shape)
    self.state_shape = np.prod(env.observation_space.shape)

  def copy_state(self, source_env):
    # self._obs_rms = copy.deepcopy(source_env._obs_rms)
    self._obs_var = copy.deepcopy(source_env._obs_var)
    self._obs_mean = copy.deepcopy(source_env._obs_mean)

  def observation(self, observation):
    if self.training:
      self._obs_normalizer.update_estimate(
        observation[..., :self.state_shape]
      )
    img_obs = observation[..., self.state_shape:]
    return np.hstack([
      self._obs_normalizer.filt(observation[..., :self.state_shape]),
      img_obs
    ])


def get_single_env(env_id, env_param):
  print(env_id, env_param)
  env = make_env(env_id, env_param["env_build"])
  env = BaseWrapper(env)

  if "rew_norm" in env_param:
    env = NormRet(env, **env_param["rew_norm"])
    del env_param["rew_norm"]

  if env_id in TIMELIMIT_DICT:
    # env = HackTimeLimit(env=env, max_episode_steps=TIMELIMIT_DICT[env_id])
    if "horizon" not in env_param:
      env = TimeLimit(env=env, max_episode_steps=TIMELIMIT_DICT[env_id])
    else:
      env = TimeLimit(env=env, max_episode_steps=env_param["horizon"])

  act_space = env.action_space
  if isinstance(act_space, gym.spaces.Box):
    env = NormAct(env)
  return env


def get_env(env_id, env_param):
  env = get_single_env(env_id, env_param)
  if "obs_norm" in env_param and env_param["obs_norm"]:
    if "get_image" in env_param["env_build"]:
      env = NormObsWithImg(env)
    else:
      env = NormObs(env)

  return env


def get_vec_env(env_id, env_param, vec_env_nums):
  if isinstance(env_param, list):
    assert vec_env_nums % len(env_param) == 0
    env_args = [
      [env_id, env_sub_params] for env_sub_params in env_param
    ] * (vec_env_nums // len(env_param))

    vec_env = VecEnv(
      vec_env_nums, [get_single_env] * vec_env_nums,
      env_args)

    if "obs_norm" in env_param[0] and env_param[0]["obs_norm"]:
      if "get_image" in env_param[0]["env_build"]:
        vec_env = NormObsWithImg(vec_env)
      else:
        vec_env = NormObs(vec_env)
    return vec_env
  else:
    vec_env = VecEnv(
      vec_env_nums, get_single_env,
      [env_id, env_param])

    if "obs_norm" in env_param and env_param["obs_norm"]:
      if "get_image" in env_param["env_build"]:
        vec_env = NormObsWithImg(vec_env)
      else:
        vec_env = NormObs(vec_env)
    return vec_env


def get_subprocvec_env(env_id, env_param, vec_env_nums, proc_nums):
  if isinstance(env_param, list):
    assert vec_env_nums % len(env_param) == 0
    env_args = [
      [env_id, env_sub_params] for env_sub_params in env_param
    ] * (vec_env_nums // len(env_param))
    vec_env = SubProcVecEnv(
      proc_nums, vec_env_nums, [get_single_env] * vec_env_nums,
      env_args
    )

    if "obs_norm" in env_param[0] and env_param[0]["obs_norm"]:
      if "get_image" in env_param[0]["env_build"]:
        vec_env = NormObsWithImg(vec_env)
      else:
        vec_env = NormObs(vec_env)
    return vec_env

  else:
    vec_env = SubProcVecEnv(
      proc_nums, vec_env_nums, get_single_env,
      [env_id, env_param])

    if "obs_norm" in env_param and env_param["obs_norm"]:
      if "get_image" in env_param["env_build"]:
        vec_env = NormObsWithImg(vec_env)
      else:
        vec_env = NormObs(vec_env)
    return vec_env
