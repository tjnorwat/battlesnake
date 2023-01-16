from render import GetNewestModel
from SnakeEnvironment import Snake
import time

game_dict = {
    "game": {
        "id": "87a650e1-c957-4033-b081-bdfe6bf1c5b8",
        "ruleset": {
            "name": "standard",
            "version": "v1.1.20",
            "settings": {
                "foodSpawnChance": 15,
                "minimumFood": 1,
                "hazardDamagePerTurn": 14,
                "hazardMap": "",
                "hazardMapAuthor": "",
                "royale": {"shrinkEveryNTurns": 0},
                "squad": {
                    "allowBodyCollisions": False,
                    "sharedElimination": False,
                    "sharedHealth": False,
                    "sharedLength": False,
                },
            },
        },
        "map": "standard",
        "timeout": 500,
        "source": "custom",
    },
    "turn": 0,
    "board": {
        "height": 7,
        "width": 7,
        "snakes": [
            {
                "id": "gs_rrkGwBcwf4qTDfJSRGBGp8Cf",
                "name": "testing_snake",
                "latency": "",
                "health": 100,
                "body": [{"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}],
                "head": {"x": 1, "y": 1},
                "length": 3,
                "shout": "",
                "squad": "",
                "customizations": {
                    "color": "#888888",
                    "head": "default",
                    "tail": "default",
                },
            }
        ],
        "food": [{"x": 0, "y": 2}, {"x": 3, "y": 3}],
        "hazards": [],
    },
    "you": {
        "id": "gs_rrkGwBcwf4qTDfJSRGBGp8Cf",
        "name": "testing_snake",
        "latency": "",
        "health": 100,
        "body": [{"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}],
        "head": {"x": 1, "y": 1},
        "length": 3,
        "shout": "",
        "squad": "",
        "customizations": {"color": "#888888", "head": "default", "tail": "default"},
    },
}


food = game_dict['board']['food']

food_positions = [ [list(val.values())[0], list(val.values())[1]] for val in food] 


env = Snake(size=7, time_between_moves=100)
env.reset()

obs = env.reset()
env.render(renderer=100)
model = GetNewestModel(env=env, recent_timestep=1668674934)
i = 0

print('prev apples', env.apple_positions)
env.SetApplePosition(food_positions)
print('new apples', env.apple_positions)

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # print(rewards)
    env.render(renderer=100)


    action = env.returnActionFromDirection()
    print('action', action)
    time.sleep(4)

    if dones:
        i += 1
        if i % 5 == 0: 
            model = GetNewestModel(env=env, recent_timestep=1668674934)

        obs = env.reset()

