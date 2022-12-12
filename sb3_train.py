from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from SnakeEnvironment import Snake
from single_instance_env import SB3SingleInstanceEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
import os

def GetNewestModel(env, recent_timestep=0, recent_file=0):

    if not recent_timestep:
        for f in os.scandir('models'):
            f = int(os.path.splitext(f.name)[0])
            if recent_timestep < f:
                recent_timestep =f


    # recent_timestep = 1662742597

    # size 6; completes the game 
    # recent_timestep = 1662781705 
    print('timestep', recent_timestep)
    models_dir = f'models/{recent_timestep}'

    if not recent_file:
        for f in os.scandir(models_dir):
            f = int(os.path.splitext(f.name)[0])
            if recent_file < f:
                recent_file = f

    print(f'zip file {recent_file}')
    model_path = f'{models_dir}/{recent_file}'

    return PPO.load(model_path, env=env, device='cpu', custom_objects=dict(n_envs=2))


snake = Snake(num_players=2)

env = SB3SingleInstanceEnv(snake)
# env = VecMonitor(env)


logdir = 'logs/out'
models_dir = 'models/'

model = PPO(
    MlpPolicy,
    env,
    tensorboard_log=logdir,
    device="cpu"                
)

# TIMESTEPS = 50_000
# iters = 0
# while True:
#     iters += 1
#     # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#     model.save(f'{models_dir}/{TIMESTEPS*iters}')


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