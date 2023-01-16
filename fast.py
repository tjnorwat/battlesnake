from SnakeEnvironment import Snake
from tqdm import tqdm


env = Snake(num_players=2)
env.reset()

for _ in tqdm(range(100_000)):
    obs, rewards, done, info = env.step([1, 1])
    if done:
        env.reset()
