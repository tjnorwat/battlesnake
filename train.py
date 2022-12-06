import os
import time
import argparse
from Snake import Snake
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


def single(size):

	env = Snake(size=size)
	return env


def multi(size):

	def main():

		def _init():
			env = Snake(size=size)
			return env

		return _init

	num_cpu = 12
	return DummyVecEnv([main() for _ in range(num_cpu)])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-s', '--single', action='store_true')
	parser.add_argument('-m', '--multi', action='store_true')
	parser.add_argument('--size', type=int, default=7)

	args = parser.parse_args()
	if args.multi:
		env = multi(args.size)
	else:
		env = single(args.size)


	model = PPO('MlpPolicy', 
	env=env, 
	verbose=1, 
	tensorboard_log=logdir, 
	device='cpu',
	batch_size=512
	)

	TIMESTEPS = 15_000
	iters = 0
	while True:
		iters += 1
		# model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
		model.save(f'{models_dir}/{TIMESTEPS*iters}')