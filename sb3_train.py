from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from SnakeEnvironment import Snake
from single_instance_env import SB3SingleInstanceEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan,  DummyVecEnv
import os
import time

# def multi(size):

# 	def main():

# 		def _init():
#             a = 1
#             snake = Snake(num_players=2)
#             env = SB3SingleInstanceEnv(snake)
#             env = VecMonitor(env)
# 			return env

# 		return _init

# 	num_cpu = 12
# 	return DummyVecEnv([main() for _ in range(num_cpu)])


def multi():

    def main():
        def _init():
            snake = Snake(num_players=2)
            env = SB3SingleInstanceEnv(snake)
            return env

        return _init

    num_envs = 2
    return DummyVecEnv([main() for _ in range(num_envs)])



snake = Snake(num_players=2)

env = SB3SingleInstanceEnv(snake)
env = VecMonitor(env)


# env = multi()

time_now = int(time.time())
models_dir = f'models/{time_now}'
logdir = f'logs/{time_now}'

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log=logdir,
    device="cpu",
    batch_size=1024
)

TIMESTEPS = 25_000
iters = 0
while True:
    iters += 1
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f'{models_dir}/{TIMESTEPS*iters}')


model.learn(1_000)

# model.save(models_dir)
# model = GetNewestModel()


obs = snake.reset()
snake.render()
while True:
    actions, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = snake.step(actions)
    snake.render()
    if done:
        obs = snake.reset()