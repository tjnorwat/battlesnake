from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Tuple


class Actions():
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class PlayerData():
    def __init__(self, ID, is_human=False, size=7):

        # left/right/up/down
        # dont need down, using it just for human
        self.direction_arr = [
            [-1, 1, -1, 1],
            [1, -1, 1, -1],
            [-1, 1, 1, -1],
            [1, -1, -1, 1]
        ]

        # x = 0, y = 1 for coordinate system
        self.axis_arr = [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]

        # left/right/up/down ; for sight
        # leftUP/leftDOWN/rightUP/rightDOWN
        self.directions = [ [-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1] ]

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
        self.score = 3 # also known as length
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
        return [ self.snake_position[0][0], self.snake_position[0][1] ]

    def getTail(self) -> list:
        return [ self.snake_position[-1][0], self.snake_position[-1][1] ]

    def getID(self) -> int:
        return self.ID

    def getScore(self) -> int:
        return self.score

    def setScore(self, score:int):
        self.score=score

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
            return -.3, None

        snake_head = [ self.snake_position[0][0], self.snake_position[0][1] ]

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
            reward = (1 - (self.moves_to_get_apple / 100))

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
        elif snake_head[0] >= self.size or \
            snake_head[0] < 0 or \
            snake_head[1] >= self.size or \
            snake_head[1] < 0:

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
            snake_head = [ self.snake_position[0][0], self.snake_position[0][1] ]

            for idx, pos in enumerate(other_snake.getPosition()):
                if snake_head == pos:
                    # both heads collide 
                    if idx == 0:
                        return True, True

                    else:
                        return True, False

        # if snake collided, collided with head of other snake
        return False, False


    def getOBS(self, snake_players: List[PlayerData], apple_positions: List[list]) -> list:

        snake_head = self.getHead()

        sight_obs = list()
        # looking in 8 directions 
        # for each direction we can see distances from  
        # * apple 
        # * wall
        # * itself
        # * another snakes * how many players there are
        # as well as if we are looking at a snakes head/tail
        for direction in self.directions:
            sight_obs += self._lookInDirection(direction, snake_head, snake_players, apple_positions)
         
        num_alive_snakes = 0


        for player in snake_players:
            if not player.isDone():
                num_alive_snakes += 1
            
            # seeing if a snake is the longest on the board
            if player.getID() != self.getID():
                if self.getScore() > player.getScore():
                    self.is_biggest_snake = 1
                else:
                    self.is_biggest_snake = 0


        norm_head = self.normalize(self.getHead(), 0, self.size - 1)
        norm_tail = self.normalize(self.getTail(), 0, self.size - 1)
        norm_direction = self.normalize_val(self.direction, 0, 3)
        norm_score = self.normalize_val(self.score, 3, self.size ** 2 - 3 * len(snake_players)) # snake length
        norm_health = self.normalize_val(self.health, 0, 100)
        norm_num_apples_on_board = self.normalize_val(len(apple_positions), 1, self.size ** 2 - 3 * len(snake_players)) # always will be 1 apple 
        norm_alive_snakes = self.normalize_val(num_alive_snakes, 0, len(snake_players)) # all snakes can die on a single turn

        normalized_obs = norm_head + \
            norm_tail + \
            [norm_direction,
            int(self.done),
            norm_score,
            norm_health,
            self.just_eat_apple,
            norm_num_apples_on_board,
            norm_alive_snakes,
            self.is_biggest_snake
            ] + sight_obs + \
            list(self.prev_actions)

        return normalized_obs


    # direction should be -1/+1 in tuple (1, -1) for x, y
    def _lookInDirection(self, direction: int, snake_head: list, other_snakes: List[PlayerData], apple_positions: List[list]) -> list:
        
        look = [-1] * (4 + len(other_snakes))
        food_found = False

        # need to check the first time we come across a specific snake
        other_snakes_found = [False] * (len(other_snakes))

        distance = 0

        # go the to position we need in the direction
        curr_pos = [snake_head[0] + direction[0], snake_head[1] + direction[1]]
        

        while not self._isCollidingWall(curr_pos):
            norm_distance = self.normalize_val(distance, -1, self.size)
            if not food_found and self._isCollidingApple(curr_pos, apple_positions):
                food_found = True
                look[0] = norm_distance
            
            # looking for other snakes
            # start index at 4 bc of look
            counter = 0
            idx = 4
            snakes_found_idx = 0 
            for other_snake in other_snakes:
                
                # making sure this isnt the same snake
                if self.getID() != other_snake.getID():
                    # seeing if we collide with another snake and have not seen it yet
                    if not other_snakes_found[snakes_found_idx] and curr_pos in other_snake.getPosition():
                        look[idx - 1 - counter] = norm_distance 
                        
                        # See if we are looking at the head/tail/body
                        # head = 1 
                        # tail = 0
                        # body/neither = -1
                        if curr_pos == other_snake.getPosition()[0]:
                            look[idx - counter] = 1
                        elif curr_pos == other_snake.getPosition()[-1]:
                            look[idx - counter] = 0

                        other_snakes_found[snakes_found_idx] = True
                        
                # if we are the same snake, dont count head cause we cant
                # current snake will always be in position 2
                elif self.getID() == other_snake.getID():
                    if curr_pos in other_snake.getPosition()[1:]:
                        # distance for this snake body 
                        look[2] = norm_distance

                        # if what we are looking at is the tail 
                        # tail = 1
                        # body/head?/nothing = -1
                        if curr_pos == other_snake.getPosition()[-1]:
                            look[3] = 1

                        other_snakes_found[snakes_found_idx] = True
                
                idx += 2
                counter += 1
                snakes_found_idx += 1

            # increment the position 
            curr_pos[0] += direction[0]
            curr_pos[1] += direction[1]

            distance += 1

        norm_distance = self.normalize_val(distance, -1, self.size)
        # wall distance
        look[1] = norm_distance

        return look


    def _isCollidingWall(self, position: list) -> bool:
        if position[0] >= self.size or \
            position[0] < 0 or \
            position[1] >= self.size or \
            position[1] < 0:

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
            temp = (((i - min_val)*diff)/diff_arr) + self.t_min
            norm_arr.append(temp)
        return norm_arr

    def normalize_val(self, val: int, min_val: int, max_val: int) -> int:
        diff = self.t_max - self.t_min
        diff_val = max_val - min_val
        return (((val - min_val)*diff)/diff_val) + self.t_min
