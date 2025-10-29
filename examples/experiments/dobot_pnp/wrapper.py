from dobot_env.envs.dobot_env import DobotEnv

class PnpEnv(DobotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
