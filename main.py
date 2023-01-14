import typing
from SnakeEnvironment import Snake
from render import GetNewestModel


env = Snake(size=7, num_players=2)
model = GetNewestModel(env=env)


def setFoodAndSnake(game_state):
    food = game_state['board']['food']
    food_positions = [ [list(val.values())[0], list(val.values())[1]] for val in food]

    snakes = game_state['board']['snakes']
    snake_positions = [[ [list(val.values())[0], list(val.values())[1]] for val in snake['body']] for snake in snakes]
    snake_lengths = [snake['length'] for snake in snakes]

    snake_ids = [snake['id'] for snake in game_state['board']['snakes']]

    # this_snake_pos = [ [list(val.values())[0], list(val.values())[1]] for val in game_state['you']['body'] ]
    this_snake_id = game_state['you']['id']
    # make sure that our snake is always the first snake
    if this_snake_id != snake_ids[0]:
        snake_positions[0], snake_positions[1] = snake_positions[1], snake_positions[0]
        snake_lengths[0], snake_lengths[1] = snake_lengths[1], snake_lengths[0]


    env.SetApplePosition(food_positions)
    env.SetSnakePosition(snake_positions)
    env.SetLength(snake_lengths)


def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Villager #4",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    env.reset() # do i even need to call reset? 
    setFoodAndSnake(game_state)
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    setFoodAndSnake(game_state)
    obs = env._GetOBS()
    actions, _states = model.predict(obs, deterministic=True)

    _, _, _, _ = env.step(actions)

    # we can guarantee the our snake is the first snake 
    next_move = env.returnActionFromDirection(player_idx=0)
    # print('move', next_move)
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})