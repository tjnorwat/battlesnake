import time
from stable_baselines3 import PPO
from SnakeEnvironment import Snake
from stable_baselines3.ppo import MlpPolicy
from multi_instance_env import SB3MultiInstanceEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan


def getGame():
    return Snake(num_players=2, size=7)


env = SB3MultiInstanceEnv(getGame, 16)
# env = VecMonitor(env)
# env = VecNormalize(env, norm_obs=False) # ?

time_now = int(time.time())

models_dir = f'models/{time_now}'
logdir = f'logs/{time_now}'


policy_kwargs={
    'net_arch': [256, 256, dict(pi=[256, 256], vf=[256, 256])]
    }    

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    batch_size=2048,
    device="cpu",
    tensorboard_log=logdir,
    policy_kwargs=policy_kwargs
)

callback = CheckpointCallback(round(50_000 / env.num_envs), save_path=models_dir, save_vecnormalize=False)
# callback = EvalCallback(env, best_model_save_path='./best_models/', log_path='./logs', eval_freq=10_000, deterministic=True, render=False)
while True:
    model.learn(50_000, callback=callback, reset_num_timesteps=False)


# another way to save models 

# TIMESTEPS = 25_000
# # TIMESTEPS = 100_000
# iters = 0
# while True:
#     iters += 1
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#     model.save(f'{models_dir}/{TIMESTEPS*iters}')
