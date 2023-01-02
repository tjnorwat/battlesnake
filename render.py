import os
import time
import argparse
from SnakeEnvironment import Snake
from stable_baselines3 import PPO
from single_instance_env import SB3SingleInstanceEnv
from multi_instance_env import SB3MultiInstanceEnv


def GetNewestModel(env, recent_timestep=0, recent_file=0):

    if not recent_timestep:
        for f in os.scandir('models'):
            f = int(os.path.splitext(f.name)[0])
            if recent_timestep < f:
                recent_timestep =f


    print('timestep', recent_timestep)
    models_dir = f'models/{recent_timestep}'
    # models_dir = f'models/'

    if not recent_file:
        for f in os.scandir(models_dir):
            f = int(os.path.splitext(f.name)[0])
            if recent_file < f:
                recent_file = f

    print(f'zip file {recent_file}')
    model_path = f'{models_dir}/{recent_file}'
    # print('model path', model_path)
    try:
        return PPO.load(model_path, env=env, device='cuda')
    except:
        return GetNewestModel(env, recent_timestep, recent_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=7)
    parser.add_argument('-t','--timestep', type=int, default=0)
    args = parser.parse_args()

    snake = Snake(num_players=2)

    env = SB3SingleInstanceEnv(snake)
    model = GetNewestModel(env=env, recent_timestep=args.timestep)

    # def getGame():
    #     return Snake(num_players=2)

    # env = SB3MultiInstanceEnv(getGame, 1)
    # model = GetNewestModel(env=env, recent_timestep=args.timestep)



    env = Snake(num_players=2, time_between_moves=200)
    obs = env.reset()
    env.render(renderer=100)
    i = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        # print(rewards)
        env.render(renderer=100)
        if done:
            i += 1
            if i % 5 == 0: 
                model = GetNewestModel(env=env, recent_timestep=args.timestep)
            time.sleep(1.5)
            # if info['won']:
            #     with open('dones.txt', 'a+') as f:
            #         f.writelines(f'GAME COMPLETED AT {time.strftime("%b %d %Y %I:%M %p")} WITH SIZE {args.size}\n')

            obs = env.reset()