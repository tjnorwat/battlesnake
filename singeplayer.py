from cv2 import waitKey
from PlayerData import Actions
from SnakeEnvironment import Snake


num_players = 2

player = Snake(size=7, is_human=True, time_between_moves=1, num_players=num_players)
player.reset()
player.render(renderer=100)

while True:

    actions = list()
    for _ in range(num_players):
        key_press = waitKey(0)

        if key_press == ord("a"):
            action = Actions.LEFT
        elif key_press == ord("d"):
            action = Actions.RIGHT
        elif key_press == ord("w"):
            action = Actions.UP
        elif key_press == ord("s"):
            action = Actions.DOWN

        actions.append(action)

    obs, rewards, done, info = player.step(actions)
    player.render(renderer=100)

    if done:
        break
