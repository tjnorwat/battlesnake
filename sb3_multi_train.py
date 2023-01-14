from SnakeEnvironment import Snake
from sb3_contrib import RecurrentPPO
from multi_instance_env import SB3MultiInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import time
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

def getGame():

    def _init():

        return Snake(num_players=2)
    
    return _init


def getGame2():
    return Snake(num_players=2, size=7)


env = SB3MultiInstanceEnv(getGame2, 16)
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
    batch_size=4096,
    n_steps=4096,
    n_epochs=10,
    vf_coef=1,
    ent_coef=.01,
    tensorboard_log=logdir,
    device="cuda",
    policy_kwargs=policy_kwargs
)


# model = RecurrentPPO(
#     'MlpLstmPolicy',
#     env,
#     verbose=1,
#     batch_size=2048,
#     tensorboard_log=logdir,
#     device='cuda'
#     )

# callback = CheckpointCallback(round(50_000 / env.num_envs), save_path=models_dir, save_vecnormalize=False)
# # callback = EvalCallback(env, best_model_save_path='./best_models/', log_path='./logs', eval_freq=10_000, deterministic=True, render=False)
# while True:
#     model.learn(50_000, callback=callback, reset_num_timesteps=False)



TIMESTEPS = 25_000
# TIMESTEPS = 100_000
iters = 0
while True:
    iters += 1
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f'{models_dir}/{TIMESTEPS*iters}')
