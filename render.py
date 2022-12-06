import os
import time
import argparse
from Snake import Snake
from stable_baselines3 import PPO


def GetNewestModel(env, recent_timestep=0, recent_file=0):

    if not recent_timestep:
        for f in os.scandir('models'):
            f = int(os.path.splitext(f.name)[0])
            if recent_timestep < f:
                recent_timestep =f


    print('timestep', recent_timestep)
    models_dir = f'models/{recent_timestep}'

    if not recent_file:
        for f in os.scandir(models_dir):
            f = int(os.path.splitext(f.name)[0])
            if recent_file < f:
                recent_file = f

    print(f'zip file {recent_file}')
    model_path = f'{models_dir}/{recent_file}'

    return PPO.load(model_path, env=env, device='cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=7)
    parser.add_argument('-t','--timestep', type=int, default=0)
    args = parser.parse_args()


    env = Snake(size=args.size, time_between_moves=100)
    env.reset()

    obs = env.reset()
    env.render(renderer=100)
    model = GetNewestModel(env=env, recent_timestep=args.timestep)
    i = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        # print(rewards)
        env.render(renderer=100)
        if dones:
            i += 1
            if i % 5 == 0: 
                model = GetNewestModel(env=env, recent_timestep=args.timestep)
            
            if info['won']:
                with open('dones.txt', 'a+') as f:
                    f.writelines(f'GAME COMPLETED AT {time.strftime("%b %d %Y %I:%M %p")} WITH SIZE {args.size}\n')

            obs = env.reset()