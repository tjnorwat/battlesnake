import gym
import numpy as np
import os
import glob
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional, List, Union, Any, Sequence, Type
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


class SB3MultiInstanceEnv(DummyVecEnv):

    def __init__(self, env_fns, num_envs):

        self.envs = [env_fns() for _ in range(num_envs)]

        self.n_agents_per_env = [m.num_players for m in self.envs]
        self.num_envs = sum(self.n_agents_per_env)

        observation_space, action_space = self.envs[0].observation_space, self.envs[0].action_space

        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def reset(self) -> VecEnvObs:

        flat_obs = list()
        for env in self.envs:
            obs = env.reset()
            flat_obs += obs

        return np.asarray(flat_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        flat_obs = list()
        flat_rews = list()
        flat_dones = list()
        flat_infos = list()

        i = 0
        for env, n_agents in zip(self.envs, self.n_agents_per_env):
            obs, reward, done, info = env.step(self.actions[i : i + n_agents])
            i += n_agents

            if done:
                info["terminal_observation"] = obs
                obs = env.reset()

            flat_obs += obs
            flat_rews += reward
            flat_dones += [done] * n_agents
            flat_infos += [info] * n_agents

        return np.asarray(flat_obs), np.array(flat_rews), np.array(flat_dones), flat_infos


class SB3SelfPlayEnv(DummyVecEnv):
    def __init__(self, env_fns, num_envs, models_dir):

        self.envs = [env_fns() for _ in range(num_envs)]

        # We only expose ONE agent per environment to the trainer
        self.num_envs = num_envs

        # Assuming all agents have same space
        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

        self.models_dir = models_dir
        self.opponent_model = None
        self.opponent_obs = [None] * num_envs

        # Try to load an initial opponent
        self.update_opponent_model()

    def update_opponent_model(self):
        """Pick a random previous model to act as opponent"""
        try:
            # Look recursively for zip files
            model_files = glob.glob(os.path.join(self.models_dir, "**/*.zip"), recursive=True)
            # Add searching in parent models dir if models_dir is a specific timestamp
            parent_dir = os.path.dirname(self.models_dir)
            if parent_dir and os.path.exists(parent_dir):
                model_files += glob.glob(os.path.join(parent_dir, "**/*.zip"), recursive=True)

            if model_files:
                random_model_path = random.choice(model_files)
                # print(f"Loading opponent: {random_model_path}")
                # Use custom_objects to map to cpu if needed, usually auto is fine
                self.opponent_model = PPO.load(random_model_path, device="cpu")
            else:
                print("No opponent models found. Opponent will play randomly.")
                self.opponent_model = None
        except Exception as e:
            print(f"Failed to load opponent: {e}")
            self.opponent_model = None

    def reset(self) -> VecEnvObs:
        obs_p1_list = []
        for i, env in enumerate(self.envs):
            # Returns [obs_p1, obs_p2]
            obs_list = env.reset()

            # Helper to manage 2 players
            # P1 (Trainer) = Index 0
            # P2 (Opponent) = Index 1

            obs_p1 = obs_list[0]
            obs_p2 = obs_list[1]

            self.opponent_obs[i] = obs_p2
            obs_p1_list.append(obs_p1)

        return np.asarray(obs_p1_list)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:

        # 1. Predict Opponent Actions
        opponent_actions = []

        if self.opponent_model is not None:
            # Stack opponent observations -> (N_Envs, H, W, C)
            # Use predict
            flat_opp_obs = np.array(self.opponent_obs)
            # PPO predict returns (actions, states)
            opp_actions_pred, _ = self.opponent_model.predict(flat_opp_obs, deterministic=True)
            opponent_actions = opp_actions_pred
        else:
            # Random actions
            for env in self.envs:
                opponent_actions.append(env.action_space.sample())

        # 2. Step Environments
        batch_obs = []
        batch_rews = []
        batch_dones = []
        batch_infos = []

        for i, env in enumerate(self.envs):
            p1_action = self.actions[i]
            p2_action = opponent_actions[i]

            # Combine actions for the env
            # env.step expects [action_p1, action_p2]
            full_actions = [p1_action, p2_action]

            # Step
            # Returns [obs1, obs2], [rew1, rew2], done, info
            obs_list, rewards_list, done, info = env.step(full_actions)

            # Handle Done / Auto-Reset
            if done:
                info["terminal_observation"] = obs_list[0]

                # Reset
                obs_list = env.reset()

            # Store P2 obs for next turn
            self.opponent_obs[i] = obs_list[1]

            # Collect P1 data for return
            batch_obs.append(obs_list[0])
            batch_rews.append(rewards_list[0])  # ONLY P1 reward
            batch_dones.append(done)
            batch_infos.append(info)

        return np.asarray(batch_obs), np.array(batch_rews), np.array(batch_dones), batch_infos

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
