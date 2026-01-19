import numpy as np
import os
import glob
import time
import argparse
from stable_baselines3 import PPO
from SnakeEnvironment import Snake
from multi_instance_env import SB3MultiInstanceEnv


def GetNewestModel(env, recent_timestep=0, recent_file=0, device="cuda"):

    if not recent_timestep:
        # Check if models directory exists
        if not os.path.exists("models"):
            print("No models directory found.")
            return None

        for f in os.scandir("models"):
            if f.is_dir():
                try:
                    f_val = int(os.path.splitext(f.name)[0])
                    if recent_timestep < f_val:
                        recent_timestep = f_val
                except ValueError:
                    continue

    print("timestep", recent_timestep)
    models_dir = f"models/{recent_timestep}"

    if not recent_file:
        list_of_files = glob.glob(models_dir + "/*.zip")
        if not list_of_files:
            print("No zip files found in model directory.")
            return None
        recent_file = max(list_of_files, key=os.path.getctime)

    print(f"zip file {recent_file}")
    try:
        return PPO.load(recent_file, env=env, device=device)
    # chance that zip file is in the middle of being written
    except Exception as e:
        print(e)
        time.sleep(1)
        return GetNewestModel(env, recent_timestep, recent_file, device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("-t", "--timestep", type=int, default=0)
    args = parser.parse_args()

    def getGame():
        return Snake(num_players=2)

    # Use Dummy Env for loading model structure
    env_wrapper = SB3MultiInstanceEnv(getGame, 1)

    # Load model (auto device)
    model = GetNewestModel(env=env_wrapper, recent_timestep=args.timestep, device="auto")

    if model is None:
        print("Could not load model. Exiting.")
        exit()

    # Create visual environment
    env = Snake(num_players=2, time_between_moves=100)

    # Initial Reset
    # Snake.reset() -> returns [obs_p1, obs_p2]
    obs_list = env.reset()

    # Needs to be stacked for model input (Batch=2, H, W, 1)
    obs = np.array(obs_list)

    env.render(renderer=50)
    i = 0

    while True:
        # Predict for both snakes: obs is (2, size, size, 1)
        # Returns actions for both
        action, lstm_states = model.predict(obs, deterministic=True)

        # Step takes list of actions [a1, a2]
        obs_list, rewards, done, info = env.step(action)

        # Update obs for next step
        obs = np.array(obs_list)

        env.render(renderer=50)

        if done:
            i += 1
            if i % 5 == 0:
                print("Checking for newer model...")
                model = GetNewestModel(env=env_wrapper, recent_timestep=args.timestep, device="auto")

            time.sleep(0.5)
            obs_list = env.reset()
            obs = np.array(obs_list)
