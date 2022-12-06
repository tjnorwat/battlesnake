import typing
from Snake import Snake
from render import GetNewestModel

env = Snake(size=7)
# model = GetNewestModel(env=env, recent_timestep=1668674934, recent_file=220455000)
model = GetNewestModel(env=env, recent_timestep=1668924532)

# helper function 
def setFoodAndSnake(game_state):
    # we have to set food and snake positions in our environment 

    food = game_state['board']['food']
    food_positions = [ [list(val.values())[0], list(val.values())[1]] for val in food] 

    snake = game_state['you']['body']
    snake_positions = [ [list(val.values())[0], list(val.values())[1]] for val in snake] 

    env.SetApplePosition(food_positions)
    env.SetSnakePosition(snake_positions)

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    env.reset()

    setFoodAndSnake(game_state)

    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:

    print('FOOD CHANCE', game_state['game']['ruleset']['settings']['foodSpawnChance'])
    setFoodAndSnake(game_state)
    obs = env._GetOBS()

    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)

    next_move = env.returnActionFromDirection()

    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({
        "info": info, 
        "start": start, 
         "move": move, 
        "end": end
    })
