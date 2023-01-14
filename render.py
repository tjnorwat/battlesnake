import os
import numpy as np
import time
import argparse
from sb3_contrib import RecurrentPPO
from SnakeEnvironment import Snake
from stable_baselines3 import PPO
from single_instance_env import SB3SingleInstanceEnv
from multi_instance_env import SB3MultiInstanceEnv
import glob


def GetNewestModel(env, recent_timestep=0, recent_file=0, device='cuda'):

    if not recent_timestep:
        for f in os.scandir('models'):
            f = int(os.path.splitext(f.name)[0])
            if recent_timestep < f:
                recent_timestep =f


    print('timestep', recent_timestep)
    models_dir = f'models/{recent_timestep}'
    # models_dir = f'models/'

    # if not recent_file:
    #     for f in os.scandir(models_dir):
    #         f = int(os.path.splitext(f.name)[0])
    #         if recent_file < f:
    #             recent_file = f

    if not recent_file:
        list_of_files = glob.glob(models_dir+'/*.zip')
        recent_file = max(list_of_files, key=os.path.getctime)
    
    
    print(f'zip file {recent_file}')
    model_path = f'{models_dir}/{recent_file}'
    # print('model path', model_path)
    try:
        return PPO.load(recent_file, env=env, device=device)
        # return PPO.load(model_path, env=env, device=device)
        # return RecurrentPPO.load(model_path, env=env, device='cuda')
    except Exception as e:
        print(e)
        time.sleep(1)
        return GetNewestModel(env, recent_timestep, recent_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=7)
    parser.add_argument('-t','--timestep', type=int, default=0)
    args = parser.parse_args()

    # snake = Snake(num_players=2)

    # env = SB3SingleInstanceEnv(snake)
    # model = GetNewestModel(env=env, recent_timestep=args.timestep)

    def getGame():
        return Snake(num_players=2)

    env = SB3MultiInstanceEnv(getGame, 1)
    model = GetNewestModel(env=env, recent_timestep=args.timestep)



    env = Snake(num_players=2, time_between_moves=200)
    obs = env.reset()
    env.render(renderer=100)
    i = 0

    lstm_states = None
    num_envs = 2
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, deterministic=True)
        # action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, done, info = env.step(action)
        episode_starts = [done] * num_envs
        env.render(renderer=100)
        if done:
            i += 1
            if i % 5 == 0: 
                model = GetNewestModel(env=env, recent_timestep=args.timestep)
            time.sleep(1.5)
            obs = env.reset()