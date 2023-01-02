from __future__ import annotations
from typing import List, Union, Tuple, Dict, Any, Type
import numpy as np
from enum import Enum
from collections import deque


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

        # get the whole matrix for the game; used for random apple 
        self.whole_coord = np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist()

        self.snake_position = list()
        self.is_human = is_human
        self.direction = 2
        self.score = 3 # also known as length
        self.moves_to_get_apple = 0
        self.total_moves = 0
        self.just_eat_apple = 0
        self.done = False
        self.health = 100
        self.prev_actions = deque([0] * 30, maxlen=30)
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

    def setDone(self, done: bool):
        self.done = done

    def isDone(self) -> bool:
        return self.done

    # might have to return whether or not we eat apple, reward, done
    # returning whether we just ate apple for reward, if we are done (collide with self or wall), and the apple to delete if we ate one
    def MoveSnake(self, action: int, apple_positions: List[list]) -> Tuple[int, list]:
        
        if isinstance(action, list):
            action = action[0]

        # make sure not dead 
        if self.done:
            return -.3, None

        snake_head = [ self.snake_position[0][0], self.snake_position[0][1] ]

        if not self.is_human:
            # print('action playerdata', action)
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

            self.moves_to_get_apple = 0
            self.health = 100


            # should reward not be based on how many moves it takes ? 
            reward = 1
            # reward = (1 - (self.moves_to_get_apple / 100)) * .5

        # otherwise just move the snake 
        else:
            self.snake_position.insert(0, snake_head)
            if self.just_eat_apple == 1:
                self.just_eat_apple = 0
            else:
                self.snake_position.pop()

            # reward based on if we see an apple? ; not sure if this would work 
            # baseline reward for being alive
            reward = .00001

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
        # not sure if complications with other snakes=
        # think it works out somehow 
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
        # snake_tail = self.getTail()

        sight_obs = list()
        # looking in 8 directions 
        # for each direction we can see distances from  
        # * apple 
        # * wall
        # * itself
        # * another snakes * how many players there are
        for direction in self.directions:
            sight_obs += self._lookInDirection(direction, snake_head, snake_players, apple_positions)
         
        # snake_dones = [player.isDone() for player in snake_players]
        # num_alive_snakes = snake_dones.count(False)

        num_alive_snakes = 0
        for player in snake_players:
            if not player.isDone():
                num_alive_snakes += 1


        norm_head = self.normalize(self.getHead(), 0, self.size - 1)
        norm_tail = self.normalize(self.getTail(), 0, self.size - 1)
        norm_direction = self.normalize_val(self.direction, 0, 3)
        norm_score = self.normalize_val(self.score, 0, self.score ** 2)
        norm_moves_to_get_apple = self.normalize_val(self.moves_to_get_apple, 0, 100)
        norm_num_apples_on_board = self.normalize_val(len(apple_positions), 1, self.size ** 2 - 3 * len(snake_players)) # always will be 1 apple 
        norm_num_snakes_on_board = self.normalize_val(len(snake_players), 0, 2) # all snakes can die on a single  turn 
        # norm_sight = self.normalize(sight_obs, -1, self.size - 1) # 
        # norm_prev_actions = self.normalize(list(self.prev_actions), -1, 2)
        norm_alive_snakes = self.normalize_val(num_alive_snakes, 0, len(snake_players))
        
        # items that dont need to be normalized
        # self.done; self.just_eat_apple
        # will have to implement sight_obs and prev_actions; thinking about doing that whenever appending to original list

        
        normalized_obs = norm_head + \
            norm_tail + \
            [norm_direction,
            self.done,
            norm_score,
            norm_moves_to_get_apple,
            self.just_eat_apple,
            norm_num_apples_on_board,
            norm_alive_snakes,
            norm_num_snakes_on_board
            ] + sight_obs + \
            list(self.prev_actions)

        # print(normalized_obs)

        # obs = [
        #     snake_head[0],
        #     snake_head[1],
        #     snake_tail[0],
        #     snake_tail[1],
        #     self.done,
        #     self.direction,
        #     self.score,
        #     self.moves_to_get_apple,
        #     self.just_eat_apple,
        #     len(apple_positions), # how many apples are on the board 
        #     num_alive_snakes,
        #     len(snake_players) # total num of how many snakes are on board
        #    ] + sight_obs \
        #  + list(self.prev_actions)

        # for multiple snakes, might have to include a done obs
        # not sure how to render that out atm 
        # 2 players is ez cause of terminal condition


        # print('single obs len', len(obs))
        # print('sight len', len(sight_obs))

        # dir_string = ['left', 'right', 'up', 'down', 'leftUP', 'leftDOWN', 'rightUP', 'rightDOWN'] # for testing 
        # factor = 2 + len(snake_players)
        # print('ID', self.ID)
        # dir_counter = 0
        # for idx, sight in enumerate(sight_obs):
        #     if idx % factor == 0:
        #         print()
        #         print(dir_string[dir_counter])
        #         dir_counter += 1
        #         print('Apple', sight)

        #     elif idx % factor == 1:
        #         print('wall', sight)
            
        #     elif idx % factor == 2:
        #         print('itself', sight)
            
        #     elif idx % factor == 3:
        #         print('other snake', sight)

        # print()

        # normalized_obs = [1]
        return normalized_obs


    # maybe it makes sense to just incldue a list of snakes as parameters and loop through as well
    # direction should be -1/+1 in tuple (1, -1) for x, y
    def _lookInDirection(self, direction: int, snake_head: list, other_snakes: List[PlayerData], apple_positions: List[list]) -> list:
        
        # changed to 2 because body found will be calculated in the for loop
        look = [-1] * (2 + len(other_snakes))

        food_found = False

        # need to check the first time we come across a specific snake
        # minus 1 because we dont count ourselves
        other_snakes_found = [False] * (len(other_snakes))

        distance = 0

        # go the to position we need in the direction
        curr_pos = [snake_head[0] + direction[0], snake_head[1] + direction[1]]
        
        # this is for swapping positions at the end 
        # we swap to always have this snake be in a static position in the array
        this_snake_pos = -1

        while not self._isCollidingWall(curr_pos):
            norm_distance = self.normalize_val(distance, -1, self.size - 1)
            if not food_found and self._isCollidingApple(curr_pos, apple_positions):
                food_found = True
                look[0] = norm_distance
            

            # replacing this with the for loop below 
            # if not body_found and self._isCollidingSnake(curr_pos):
            #     body_found = True
            #     look[1] = distance


            # looking for other snakes
            # start index at 2 bc of look
            # can subtract 2 from other_snakes_found bc that is just for other snakes
            for idx, other_snake in enumerate(other_snakes, 2):
                # making sure this isnt the same snake
                if self.getID() != other_snake.getID():
                    # seeing if we collide with another snake and have not seen it yet
                    if not other_snakes_found[idx - 2] and curr_pos in other_snake.getPosition():
                        look[idx] = norm_distance
                        other_snakes_found[idx - 2] = True
                
                # if we are the same snake, dont count head
                # WILL HAVE TO MAKE SURE THE THIS IS IN THE SAME POSITION
                # JUST SWAP POSTIIONS AT THE END 
                elif self.getID() == other_snake.getID():
                    if not other_snakes_found[idx - 2] and curr_pos in other_snake.getPosition()[1:]:
                        look[idx] = norm_distance
                        this_snake_pos = idx
                        other_snakes_found[idx - 2] = True


            # increment the position 
            curr_pos[0] += direction[0]
            curr_pos[1] += direction[1]

            distance += 1

        norm_distance = self.normalize_val(distance, -1, self.size - 1)
        # wall distance
        look[1] = norm_distance

        # swapping pos 2 with this snakes body 
        # dont need to swap if same 
        if this_snake_pos != 2:
            look[2], look[this_snake_pos] = look[this_snake_pos], look[2]

        return look


    def _isCollidingWall(self, position: list) -> bool:
        # if we are outside of the walls
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
        # normalize values between -1 and 1 
        t_min = -1
        t_max = 1

        norm_arr = []
        diff = t_max - t_min
        diff_arr = max_val - min_val
        for i in arr:
            temp = (((i - min_val)*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr


    def normalize_val(self, val: int, min_val: int, max_val: int) -> int:
        return (((val - min_val) * 2) / max_val) + -1


    # def normalize_val(self, val: int, min_val: int, max_val: int) -> int:
    #     t_min = -1
    #     t_max = 1

    #     diff = t_max - t_min
    #     diff_val = max_val
    #     return (((val - min_val)*diff)/diff_val) + t_min


    def __str__(self):

        return str(self.snake_position)