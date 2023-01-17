import os
import glob
import time
import argparse
from stable_baselines3 import PPO
from SnakeEnvironment import Snake
from multi_instance_env import SB3MultiInstanceEnv


def GetNewestModel(env, recent_timestep=0, recent_file=0, device="cuda"):

    if not recent_timestep:
        for f in os.scandir("models"):
            f = int(os.path.splitext(f.name)[0])
            if recent_timestep < f:
                recent_timestep = f

    print("timestep", recent_timestep)
    models_dir = f"models/{recent_timestep}"

    if not recent_file:
        list_of_files = glob.glob(models_dir + "/*.zip")
        recent_file = max(list_of_files, key=os.path.getctime)

    print(f"zip file {recent_file}")
    try:
        return PPO.load(recent_file, env=env, device=device)
    # chance that zip file is in the middle of being written
    except Exception as e:
        print(e)
        time.sleep(1)
        return GetNewestModel(env, recent_timestep, recent_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("-t", "--timestep", type=int, default=0)
    args = parser.parse_args()

    def getGame():
        return Snake(num_players=2)

    env = SB3MultiInstanceEnv(getGame, 1)
    model = GetNewestModel(env=env, recent_timestep=args.timestep)

    env = Snake(num_players=2, time_between_moves=200)
    obs = env.reset()
    env.render(renderer=100)
    i = 0

    while True:
        action, lstm_states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render(renderer=100)
        if done:
            i += 1
            if i % 5 == 0:
                model = GetNewestModel(env=env, recent_timestep=args.timestep)
            time.sleep(1.5)
            obs = env.reset()
