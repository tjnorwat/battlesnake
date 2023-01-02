import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional, List, Union, Any, Sequence, Type
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


class SB3MultiInstanceEnv(DummyVecEnv):

    def __init__(self, env_fns, num_envs):

        self.envs = [env_fns() for _ in range(num_envs)]
        
        # print(self.envs[0])
        self.n_agents_per_env = [m.num_players for m in self.envs]
        self.num_envs = sum(self.n_agents_per_env) 

        # print('num_envs', self.num_envs)
        observation_space, action_space = self.envs[0].observation_space, self.envs[0].action_space

        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

        # super().__init__(envs)


    def reset(self) -> VecEnvObs:
        
        flat_obs = list()
        for env in self.envs:
            obs = env.reset()
            flat_obs += obs


        # flat_obs = [[1,2],[3,4]] * 12
        # print('IN RESET', flat_obs)
        # print('shape', np.shape(flat_obs))
        return np.array(flat_obs)


    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        # print('action size', np.shape(actions))

    def step_wait(self) -> VecEnvStepReturn:
        # print('ohaehay')
        flat_obs = list()
        flat_rews = list()
        flat_dones = list()
        flat_infos = list()

        i = 0
        for env, n_agents in zip(self.envs, self.n_agents_per_env):
            obs, reward, done, info = env.step(self.actions[i:i+n_agents])
            i += n_agents

            if done:
                info['terminal_observation'] = obs
                obs = env.reset()

            flat_obs += obs
            flat_rews += reward
            flat_dones += [done] * n_agents
            flat_infos += [info] * n_agents

        # flat_obs = [[1,2],[3,4]] * 12 
        # print(flat_obs)
        # print(np.shape(flat_obs))
        # exit()
        return np.array(flat_obs), np.array(flat_rews), np.array(flat_dones), flat_infos 



    # # i think the goal of this is to send actions to each instance 

    # # observation shape needs to be half ? 
    # def reset(self) -> VecEnvObs:
        
    #     flat_obs = list()

    #     for env in self.envs:
    #         obs = env.reset()
    #         # print('OBS', obs)
    #         flat_obs.append(obs)

    #     # print('shape', np.shape(flat_obs))
    #     return np.array(flat_obs)

    # def step_async(self, actions: np.ndarray) -> None:
    #     self.actions = actions


    # def step_wait(self) -> VecEnvStepReturn:
    #     flat_obs = list()
    #     flat_rews = list()
    #     flat_dones = list()
    #     flat_infos = list()

    #     i = 0
    #     for env, n_agents in zip(self.envs, self.n_agents_per_env):
    #         obs, rew, done, info = env.step(self.actions[i : i + n_agents])

    #         print('OBSERVATION', obs)
    #         print('REWARDS', rew)
    #         print('DONE', done)
    #         print('INFOS', info)

    #         # flat_obs.append(obs)
    #         # flat_rews.append(rew)
    #         # flat_dones.append(done)
    #         # flat_infos.append(info)

    #         if done:
    #             infos = [info] * len(rew)
    #             for info, obs in zip(infos, obs):
    #                 info['terminal_observation'] = obs

    #             obs = env.reset()

    #         flat_obs.append(obs)
    #         flat_rews.append(rew)
    #         flat_infos.append(info)
    #         flat_dones.append(done)


    #         i += n_agents

        
    #     print('FLAT OBSERVATION', flat_obs)
    #     print()
    #     print('FLAT REWARDS', flat_rews)
    #     print()
    #     print('FLAT DONE', flat_dones)
    #     print()
    #     print('FLAT INFOS', flat_infos)
    #     print()

    #     return (np.array(flat_obs), np.array(flat_rews), np.array(flat_dones), np.array(flat_infos))


    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass
    
    def close(self) -> None:
        pass

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass
    
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        pass
    
    def get_images(self) -> Sequence[np.ndarray]:
        pass
    
    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        pass

    def _obs_from_buf(self) -> VecEnvObs:
        pass

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        pass