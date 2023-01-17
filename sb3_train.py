import time
from stable_baselines3 import PPO
from SnakeEnvironment import Snake
from stable_baselines3.ppo import MlpPolicy
from single_instance_env import SB3SingleInstanceEnv
from stable_baselines3.common.vec_env import VecMonitor

snake = Snake(num_players=2)

env = SB3SingleInstanceEnv(snake)
env = VecMonitor(env)

time_now = int(time.time())
models_dir = f'models/{time_now}'
logdir = f'logs/{time_now}'

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log=logdir,
    device="cpu",
    batch_size=2048
)

TIMESTEPS = 25_000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f'{models_dir}/{TIMESTEPS*iters}')
