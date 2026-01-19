import time
import argparse
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from SnakeEnvironment import Snake
from stable_baselines3.ppo import CnnPolicy
from multi_instance_env import SB3SelfPlayEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def getGame():
    return Snake(num_players=2, size=7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="Device to train on (auto, cpu, cuda)")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    args = parser.parse_args()

    time_now = int(time.time())
    models_dir = f"models/{time_now}"
    logdir = f"logs/{time_now}"

    # Use the new Self Play Env
    # Models dir is passed so it can find historical opponents
    env = SB3SelfPlayEnv(getGame, args.num_envs, models_dir="models")

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256]),
    }

    model = PPO(
        CnnPolicy,
        env,
        verbose=1,
        batch_size=2048,
        device=args.device,
        tensorboard_log=logdir,
        policy_kwargs=policy_kwargs,
    )

    callback = CheckpointCallback(round(50_000 / env.num_envs), save_path=models_dir, save_vecnormalize=False)

    TIMESTEPS = 50_000
    while True:
        model.learn(TIMESTEPS, callback=callback, reset_num_timesteps=False)
        # Update opponent every training cycle (or less frequently if desired)
        print("Updating opponent model...")
        env.update_opponent_model()
        CnnPolicy,
        env,
        verbose = (1,)
        batch_size = (2048,)
        device = (args.device,)
        tensorboard_log = (logdir,)
        policy_kwargs = (policy_kwargs,)

    callback = CheckpointCallback(round(50_000 / env.num_envs), save_path=models_dir, save_vecnormalize=False)

    while True:
        model.learn(50_000, callback=callback, reset_num_timesteps=False)
