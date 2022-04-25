from vision4leg.envs import env_builder as eb

SETTING_DICT = {
    "Gravity-1": "gravity_1.xml",
    "Gravity-5": "gravity_5.xml",
    "Gravity-1-bounce": "gravity_1_bounce.xml",
    "bounce": "bounce.xml",
    "default": "default.xml"
}

ENV_DICT = {
    "A1MoveGround": eb.build_a1_ground_env,
    "A1MoveGroundMPC": eb.build_a1_ground_mpc_env,
}

NO_PATH_LILST = [
    "A1MoveGround",
    "A1MoveGroundMPC",
]

TIMELIMIT_DICT = {
    "A1MoveGround": 1000,
    "A1MoveGroundMPC": 1000
}
