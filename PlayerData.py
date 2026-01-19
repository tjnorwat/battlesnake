from __future__ import annotations
import numpy as np
from collections import deque
from enum import IntEnum
from typing import List, Tuple, Set, Dict


class Actions(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class PlayerData:
    def __init__(self, ID, is_human=False, size=7):

        # left/right/up/down
        # dont need down, using it just for human
        self.direction_arr = [[-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1]]

        # x = 0, y = 1 for coordinate system
        self.axis_arr = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]

        # left/right/up/down ; for sight
        # leftUP/leftDOWN/rightUP/rightDOWN
        self.directions = [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1]]

        self.ID = ID
        self.size = size

        # normalization range
        self.t_min = -1
        self.t_max = 1

        # get the whole matrix for the game; used for random apple
        self.whole_coord = np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist()

        self.is_biggest_snake = 0
        self.snake_position = list()
        self.is_human = is_human
        self.direction = 2
        self.score = 3  # also known as length
        self.moves_to_get_apple = 0
        self.total_moves = 0
        self.just_eat_apple = 0
        self.done = False
        self.health = 100
        self.prev_actions = deque([-1] * 30, maxlen=30)
        self.max_score = 0

    def SetPosition(self, position: List[list]):
        self.snake_position = position

    def SetIsHuman(self, is_human: bool):
        self.is_human = is_human

    def AddPosition(self, position: list):
        self.snake_position.append(position)

    def getPosition(self) -> List[list]:
        return self.snake_position

    def getHead(self) -> list:
        return [self.snake_position[0][0], self.snake_position[0][1]]

    def getTail(self) -> list:
        return [self.snake_position[-1][0], self.snake_position[-1][1]]

    def getID(self) -> int:
        return self.ID

    def getScore(self) -> int:
        return self.score

    def setScore(self, score: int):
        self.score = score

    def setDone(self, done: bool):
        self.done = done

    def isDone(self) -> bool:
        return self.done

    def getHealth(self) -> bool:
        return self.health

    def setHealth(self, health):
        self.health = health

    def addToPrevAction(self, action):
        norm_action = self.normalize_val(action, 0, 2)
        self.prev_actions.append(norm_action)

    def setDirection(self, direction):
        self.direction = direction

    def getDirection(self):
        return self.direction

    def setAteApple(self, ate_apple: int):
        self.just_eat_apple = ate_apple

    def getAteApple(self):
        return self.just_eat_apple

    def setBiggestSnake(self, is_biggest: int):
        self.is_biggest_snake = is_biggest

    def getBiggestSnake(self):
        return self.is_biggest_snake

    # returning whether we just ate apple for reward and the apple to delete if we ate one
    def MoveSnake(self, action: int, apple_positions: List[list]) -> Tuple[int, list]:

        if isinstance(action, list):
            action = action[0]

        # make sure not dead
        if self.done:
            return -0.3, None

        snake_head = [self.snake_position[0][0], self.snake_position[0][1]]

        if not self.is_human:
            val = self.direction_arr[action][self.direction]
            which_axis = self.axis_arr[action][self.direction]

            # get new direction
            if val == -1 and which_axis == 0:
                self.direction = Actions.LEFT
            elif val == -1 and which_axis == 1:
                self.direction = Actions.DOWN
            elif val == 1 and which_axis == 0:
                self.direction = Actions.RIGHT
            else:
                self.direction = Actions.UP

            # move the snake head
            if which_axis == 0:
                snake_head[0] += val
            else:
                snake_head[1] += val

        else:
            # this also keeps the direction to up / doesnt change
            # left, right, up, down
            if action == Actions.LEFT:
                snake_head[0] -= 1
            elif action == Actions.RIGHT:
                snake_head[0] += 1
            # switched up and down
            elif action == Actions.UP:
                snake_head[1] += 1
            elif action == Actions.DOWN:
                snake_head[1] -= 1

        self.moves_to_get_apple += 1
        self.health -= 1
        self.total_moves += 1

        norm_action = self.normalize_val(action, 0, 2)
        self.prev_actions.append(norm_action)

        # seeing if we just ate an apple
        if snake_head in apple_positions:
            self.score += 1

            # remove the apple somehow ?? ?
            # apple_positions.remove()
            self.snake_position.insert(0, snake_head)

            if self.just_eat_apple == 0:
                self.snake_position.pop()

            self.just_eat_apple = 1

            if self.score > self.max_score:
                self.max_score = self.score

            # should reward not be based on how many moves it takes ?
            # reward = 1
            reward = 1 - (self.moves_to_get_apple / 100)

            self.moves_to_get_apple = 0
            self.health = 100

        # otherwise just move the snake
        else:
            self.snake_position.insert(0, snake_head)
            if self.just_eat_apple == 1:
                self.just_eat_apple = 0
            else:
                self.snake_position.pop()

            # reward based on if we see an apple? ; not sure if this would work
            # baseline reward for being alive
            # reward = .00001
            reward = 0

        # collisiion with self
        if self.total_moves > 3 and snake_head in self.snake_position[1:]:
            self.done = True
            reward = -3

        # collision with walls
        elif snake_head[0] >= self.size or snake_head[0] < 0 or snake_head[1] >= self.size or snake_head[1] < 0:

            self.done = True
            reward = -3

        # made too many moves
        elif self.moves_to_get_apple >= 100 or self.health <= 0:
            self.done = True
            reward = -3

        # have to send back which apple we just ate
        if self.just_eat_apple:
            apple_to_delete = snake_head
        else:
            apple_to_delete = None

        return reward, apple_to_delete

    # need to see if it is head that collided or other body part
    # because if it is head, need to check which snake is longer to see who dies
    def collideWithOtherSnakes(self, other_snake: PlayerData) -> Tuple[bool, bool]:
        if self.ID != other_snake.getID():
            snake_head = [self.snake_position[0][0], self.snake_position[0][1]]

            for idx, pos in enumerate(other_snake.getPosition()):
                if snake_head == pos:
                    # both heads collide
                    if idx == 0:
                        return True, True

                    else:
                        return True, False

        # if snake collided, collided with head of other snake
        return False, False

    def getOBS(self, snake_players: List[PlayerData], apple_positions: List[list]) -> np.ndarray:

        # Init grid (H, W, 1) for Channels Last format typical in Gym
        # Using float32 for CNN
        grid = np.zeros((self.size, self.size, 1), dtype=np.float32)

        # Place apples (Value: 0.5 - distinct from snakes)
        for apple_pos in apple_positions:
            x, y = apple_pos
            if 0 <= x < self.size and 0 <= y < self.size:
                grid[x, y, 0] = 5.0

        # Place snakes
        for player in snake_players:
            body = player.getPosition()
            length = len(body)
            if length == 0:
                continue

            is_self = player.getID() == self.getID()

            # Self:  Head=1.0  -> Tail approach 0.2
            # Enemy: Head=-1.0 -> Tail approach -0.2
            # We use a gradient to encode direction

            for i, part in enumerate(body):
                x, y = part
                if 0 <= x < self.size and 0 <= y < self.size:
                    # Normalized segment index (0.0 at head, 1.0 at tail)
                    # Avoid division by zero for length 1
                    segment_ratio = i / (length - 1) if length > 1 else 0

                    if is_self:
                        # 1.0 down to 0.2
                        val = 1.0 - (segment_ratio * 0.8)
                    else:
                        # -1.0 up to -0.2
                        val = -1.0 + (segment_ratio * 0.8)

                    grid[x, y, 0] = val

        return grid

    def _isCollidingWall(self, position: list) -> bool:
        if position[0] >= self.size or position[0] < 0 or position[1] >= self.size or position[1] < 0:

            return True

        return False

    def _isCollidingApple(self, position: list, apple_positions: List[list]):
        if position in apple_positions:
            return True

        return False

    def _isCollidingSnake(self, position: list) -> bool:
        if position in self.snake_position:
            return True

        return False

    def normalize(self, arr: list, min_val: int, max_val: int) -> list:
        norm_arr = []
        diff = self.t_max - self.t_min
        diff_arr = max_val - min_val
        for i in arr:
            temp = (((i - min_val) * diff) / diff_arr) + self.t_min
            norm_arr.append(temp)
        return norm_arr

    def normalize_val(self, val: int, min_val: int, max_val: int) -> int:
        diff = self.t_max - self.t_min
        diff_val = max_val - min_val
        return (((val - min_val) * diff) / diff_val) + self.t_min
