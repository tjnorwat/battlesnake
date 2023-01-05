from SnakeEnvironment import Snake
from multi_instance_env import SB3MultiInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import time
import torch

def getGame():

    def _init():

        return Snake(num_players=2)
    
    return _init


def getGame2():
    return Snake(num_players=2)


env = SB3MultiInstanceEnv(getGame2, 12)

time_now = int(time.time())


models_dir = f'models/{time_now}'
logdir = f'logs/{time_now}'


# model = PPO(
#     MlpPolicy,
#     env,
#     verbose=1,
#     tensorboard_log=logdir,
#     device="cpu",
#     batch_size=4096
# )

policy_kwargs={
    'activation_fn': torch.nn.ReLU,
    'net_arch': [256, 256]
    }    

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    batch_size=4096,             # Batch size as high as possible within reason
    tensorboard_log=logdir,      # `tensorboard --logdir out/logs` in terminal to see graphs
    device="cpu",
    policy_kwargs={
        'net_arch': [256, 256]
    }          
)


# model = PPO(
#     MlpPolicy,
#     env,
#     n_epochs=32,                 # PPO calls for multiple epochs
#     learning_rate=1e-5,          # Around this is fairly common for PPO
#     ent_coef=0.01,               # From PPO Atari
#     vf_coef=1.,                  # From PPO Atari
#     verbose=1,
#     batch_size=4096,             # Batch size as high as possible within reason
#     tensorboard_log=logdir,      # `tensorboard --logdir out/logs` in terminal to see graphs
#     device="cpu"                
# )





TIMESTEPS = 25_000
iters = 0
while True:
    iters += 1
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f'{models_dir}/{TIMESTEPS*iters}')
