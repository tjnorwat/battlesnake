from render import GetNewestModel
from SnakeEnvironment import Snake

# Global storage for game states
games = {}

# Global model (shared across games)
# Initialize with a dummy environment to load the model structure
dummy_env = Snake(size=11, num_players=2)
model = GetNewestModel(env=dummy_env)


def get_game_id(game_state):
    return game_state["game"]["id"]


def setFoodAndSnake(env, game_state: dict):
    food = game_state["board"]["food"]
    food_positions = [[list(val.values())[0], list(val.values())[1]] for val in food]

    snakes = game_state["board"]["snakes"]
    snake_positions = [[[list(val.values())[0], list(val.values())[1]] for val in snake["body"]] for snake in snakes]
    snake_lengths = [snake["length"] for snake in snakes]
    snake_healths = [snake["health"] for snake in game_state["board"]["snakes"]]

    snake_ids = [snake["id"] for snake in game_state["board"]["snakes"]]

    this_snake_id = game_state["you"]["id"]
    # make sure that our snake is always the first snake
    if this_snake_id != snake_ids[0]:
        snake_positions[0], snake_positions[1] = snake_positions[1], snake_positions[0]
        snake_lengths[0], snake_lengths[1] = snake_lengths[1], snake_lengths[0]
        snake_healths[0], snake_healths[1] = snake_healths[1], snake_healths[0]

    env.setApplePosition(food_positions)
    env.setSnakePosition(snake_positions)
    env.setLength(snake_lengths)
    env.setHealth(snake_healths)


def info() -> dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Villager #4",  # TODO: Your Battlesnake Username
        "color": "#6495ED",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


def start(game_state: dict):
    game_id = get_game_id(game_state)
    print(f"GAME START {game_id}")

    env = Snake(size=game_state["board"]["height"], num_players=2)
    env.reset()

    # Store game state
    games[game_id] = {"env": env, "turn": 0}

    setFoodAndSnake(env, game_state)


def end(game_state: dict):
    game_id = get_game_id(game_state)
    print(f"GAME OVER {game_id}\n")
    if game_id in games:
        del games[game_id]


def move(game_state: dict) -> dict:
    game_id = get_game_id(game_state)

    # Recover game state
    if game_id not in games:
        # Should not happen ideally, but as fallback
        start(game_state)

    game_data = games[game_id]
    env = game_data["env"]
    turn = game_data["turn"]

    # need to get the action/direction of the other snake before we call the obs

    # need this to figure out direction/action of other snake
    other_snake_prev_head = env.getPlayerHead(player_idx=1)
    prev_apple_positions = env.getApplePositions()

    # get the new updated position of the snakes/apples
    # important for snake we are versing
    setFoodAndSnake(env, game_state)

    # getting action/direction for other snake
    # do this before we get the obs
    # if this is the first turn, we cant get the previous action of the other snake bc there is none
    if turn != 0:

        curr_head = env.getPlayerHead(player_idx=1)

        x_diff = curr_head[0] - other_snake_prev_head[0]
        y_diff = curr_head[1] - other_snake_prev_head[1]

        if x_diff < 0:  # left
            action_and_direction = 0
        elif x_diff > 0:  # right
            action_and_direction = 1
        elif y_diff > 0:  # up
            action_and_direction = 2
        elif y_diff < 0:  # down
            action_and_direction = 3

        # if other snake dies first, action_and_direction var is unbound
        try:
            # "stepping" other snake
            env.addDirection(player_idx=1, direction=action_and_direction)
            env.addActionToList(player_idx=1, action=action_and_direction)
            # checking if other snake just ate apple
            if curr_head in prev_apple_positions:
                env.ateApple(player_idx=1, ate_apple=1)
            else:
                env.ateApple(player_idx=1, ate_apple=0)
        except UnboundLocalError:
            pass

    game_data["turn"] += 1

    obs = env._GetOBS()
    actions, _states = model.predict(obs, deterministic=True)

    # manually stepping just our snake
    env.stepPlayer(player_idx=0, action=actions[0])

    # we can guarantee that our snake is the first snake
    next_move = env.getActionFromDirection(player_idx=0)

    env.render(70)
    print("move", next_move)
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
