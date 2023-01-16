import typing
from SnakeEnvironment import Snake
from render import GetNewestModel


env = Snake(size=7, num_players=2)
model = GetNewestModel(env=env)
turn = 0

def setFoodAndSnake(game_state):
    food = game_state['board']['food']
    food_positions = [ [list(val.values())[0], list(val.values())[1]] for val in food]

    snakes = game_state['board']['snakes']
    snake_positions = [[ [list(val.values())[0], list(val.values())[1]] for val in snake['body']] for snake in snakes]
    snake_lengths = [snake['length'] for snake in snakes]
    snake_healths = [snake['health'] for snake in game_state['board']['snakes']]

    snake_ids = [snake['id'] for snake in game_state['board']['snakes']]

    this_snake_id = game_state['you']['id']
    # make sure that our snake is always the first snake
    if this_snake_id != snake_ids[0]:
        snake_positions[0], snake_positions[1] = snake_positions[1], snake_positions[0]
        snake_lengths[0], snake_lengths[1] = snake_lengths[1], snake_lengths[0]
        snake_healths[0], snake_healths[1] = snake_healths[1], snake_healths[0]


    env.setApplePosition(food_positions)
    env.setSnakePosition(snake_positions)
    env.setLength(snake_lengths)
    env.setHealth(snake_healths)

def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Villager #4",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


def start(game_state: typing.Dict):
    env.reset() 
    setFoodAndSnake(game_state)
    global turn
    turn = 0
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:


    # need to get the action of the other snake before we call the obs 
    # need to also change the direction of other snake?

    # need this to figure out direction/action of other snake 
    other_snake_prev_head = env.getPlayerHead(player_idx=1)
    prev_apple_positions = env.getApplePositions()
    # other_snake_prev_direction = env.getPlayerDirection(player_idx=1)

    # get the new updated position of the snakes/apples 
    # important for other snake 
    setFoodAndSnake(game_state)

    # getting action/direction for other snake 
    # do this before we get the obs 
    global turn
    # if this is the first turn, we cant get the previous action of the other snake bc there is none
    if turn != 0:
        
        # if this is the second turn, getting the direction is hard 
        # if turn == 1:
        #     # check if x goes down; this is for if the snake direction is down on first move 

        # else:

            curr_head = env.getPlayerHead(player_idx=1)

            x_diff = curr_head[0] - other_snake_prev_head[0]
            y_diff = curr_head[1] - other_snake_prev_head[1]

            if x_diff < 0: # left
                action_and_direction = 0
            elif x_diff > 0: # right
                action_and_direction = 1
            elif y_diff > 0: # up
                action_and_direction = 2
            elif y_diff < 0: # down
                action_and_direction = 3
            
            # if other snake dies first, action_and_direction var is unbound 
            try:
                # "stepping" other snake
                env.addDirection(player_idx=1, direction=action_and_direction)
                env.addActionToList(player_idx=1, action=action_and_direction)
                # checking if other snake just ate apple 
                if curr_head in prev_apple_positions:
                    env.ateApple(player_idx=1, ate_apple=1)
                    # print(f'turn {turn} ate true')
                else:
                    env.ateApple(player_idx=1, ate_apple=0)
            except UnboundLocalError:
                pass

    # print(f'turn {turn} just ate apple', env.getAteApple(player_idx=0))

    turn += 1

    obs = env._GetOBS()
    actions, _states = model.predict(obs, deterministic=True)

    # manually stepping just our snake 
    env.stepPlayer(player_idx=0, action=actions[0])


    # env.addActionToList(player_idx=0, action=actions[0])
    
    
    
    # i need this to add to the previous action list
    # _, _, _, _ = env.step(actions)


    # print('actions', actions)
    # we can guarantee that our snake is the first snake 
    next_move = env.getActionFromDirection(player_idx=0)
    print('move', next_move)
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})