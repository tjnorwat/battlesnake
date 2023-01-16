import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn, VecEnvObs

class SB3SingleInstanceEnv(VecEnv):

    def __init__(self, env):
        
        super().__init__(env.num_players, env.observation_space, env.action_space)
        self.env = env
        self.step_result = None

    def reset(self):
        observations = self.env.reset()
        # print('step_wait obs shape', np.shape(observations))
        if len(np.shape(observations)) == 1:
            observations = [observations]
        return np.asarray(observations)
        # return np.concatenate(observations)

    def step_async(self, actions: np.array) -> None:
        # print('actions from vecenv', actions)
        # return super().step_async(actions)
        self.step_result = self.env.step(actions)
        return self.step_result

    def step_wait(self) -> VecEnvStepReturn:
        # observations, rewards, done, info = super().step_wait()
        observations, rewards, done, info = self.step_result

        # observations = [observations]
        # rewards = [[reward] for reward in rewards]
        # print('rewards', rewards)
        # print('rewards shape', np.shape(rewards))

        if done:
            infos = [info] * len(rewards)
            for info, obs in zip(infos, observations):
                info['terminal_observation'] = obs
            
            observations = self.env.reset()
            if len(np.shape(observations)) == 1:
                observations = [observations]
        
        else:
            infos = [info] * len(rewards)

        return np.asarray(observations), np.asarray(rewards), np.full(len(rewards), done), infos
        # return np.concatenate(observations), rewards, np.full(len(rewards), done), info

    def close(self) -> None:
        pass

    def seed(self, seed):
        pass

    # Now a bunch of functions that need to be overridden to work, might have to implement later
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        pass

    def set_attr(self, attr_name: str, value, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs):
        return [False] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        return [False] * self.num_envs

    def get_images(self):
        pass