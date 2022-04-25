from .atari_wrapper import *
from .continuous_wrapper import *
from .base_wrapper import *
from .vecenv import VecEnv
from .subproc_vecenv import SubProcVecEnv


def wrap_deepmind(env, frame_stack=False, scale=False, clip_rewards=False):
  assert 'NoFrameskip' in env.spec.id
  env = EpisodicLifeEnv(env)
  env = NoopResetEnv(env, noop_max=30)
  env = MaxAndSkipEnv(env, skip=4)
  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
  env = WarpFrame(env)
  if scale:
    env = ScaledFloatFrame(env)
  if clip_rewards:
    env = ClipRewardEnv(env)
  if frame_stack:
    env = FrameStack(env, 4)
  return env


def wrap_continuous_env(env, obs_norm, reward_scale):
  env = RewardShift(env, reward_scale)
  if obs_norm:
    return NormObs(env)
  return env


def get_env(env_id, env_param):
  env = gym.make(env_id)
  if str(env.__class__.__name__).find('TimeLimit') >= 0:
    env = TimeLimitAugment(env)
  env = BaseWrapper(env)
  if "rew_norm" in env_param:
    env = NormRet(env, **env_param["rew_norm"])
    del env_param["rew_norm"]

  ob_space = env.observation_space
  if len(ob_space.shape) == 3:
    env = wrap_deepmind(env, **env_param)
  else:
    env = wrap_continuous_env(env, **env_param)

  if isinstance(env.action_space, gym.spaces.Box):
    return NormAct(env)
  return env


def get_single_env(env_id, env_param):
  env = gym.make(env_id)
  if str(env.__class__.__name__).find('TimeLimit') >= 0:
    env = TimeLimitAugment(env)
  env = BaseWrapper(env)

  ob_space = env.observation_space
  if len(ob_space.shape) == 3:
    env = wrap_deepmind(env, **env_param)

  if "reward_scale" in env_param:
    env = RewardShift(env, env_param["reward_scale"])

  if isinstance(env.action_space, gym.spaces.Box):
    return NormAct(env)
  return env


def get_vec_env(env_id, env_param, vec_env_nums):
  vec_env = VecEnv(
    vec_env_nums, get_single_env,
    [env_id, env_param])

  if "obs_norm" in env_param and env_param["obs_norm"]:
    vec_env = NormObs(vec_env)
  return vec_env


def get_subprocvec_env(env_id, env_param, vec_env_nums, proc_nums):
  vec_env = SubProcVecEnv(
    proc_nums, vec_env_nums, get_single_env,
    [env_id, env_param])

  if "obs_norm" in env_param and env_param["obs_norm"]:
    vec_env = NormObs(vec_env)
  return vec_env
