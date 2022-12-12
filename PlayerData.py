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
        self.prev_actions = deque([-1] * 100, maxlen=100)
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
            print('action', action)
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

        self.prev_actions.append(action)

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
            # reward = 1
            reward = (1 - (self.moves_to_get_apple / (self.size ** 2 - 1))) * .5

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
            reward = -1
        
        # collision with walls
        elif snake_head[0] >= self.size or \
            snake_head[0] < 0 or \
            snake_head[1] >= self.size or \
            snake_head[1] < 0:

            self.done = True
            reward = -1
        
        elif self.moves_to_get_apple >= 100 or self.health <= 0:
            self.done = True
            reward = -.5

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
        snake_tail = self.getTail()

        sight_obs = list()
        # looking in 8 directions 
        # for each direction we can see distances from  
        # * apple 
        # * wall
        # * itself
        # * another snakes * how many players there are
        for direction in self.directions:
            sight_obs += self._lookInDirection(direction, snake_head, snake_players, apple_positions)
         
        # TODO normalize LATER 
        obs = [
            snake_head[0],
            snake_head[1],
            snake_tail[0],
            snake_tail[1],
            self.done,
            self.direction,
            self.score,
            self.moves_to_get_apple,
            self.just_eat_apple,
            len(apple_positions), # how many apples are on the baord 
            len(snake_players) # how many snakes are on the board 
           ] + sight_obs \
         + list(self.prev_actions)

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


        return obs


    # maybe it makes sense to just incldue a list of snakes as parameters and loop through as well
    # direction should be -1/+1 in tuple (1, -1) for x, y
    def _lookInDirection(self, direction: int, snake_head: list, other_snakes: List[PlayerData], apple_positions: List[list]) -> list:
        
        # changed to 2 because body found will be calculated in the for loop
        look = [-1] * (2 + len(other_snakes))

        food_found = False
        # get rid of this because we can do it in 
        body_found = False

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

            if not food_found and self._isCollidingApple(curr_pos, apple_positions):
                food_found = True
                look[0] = distance
            

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
                        look[idx] = distance
                        other_snakes_found[idx - 2] = True
                
                # if we are the same snake, dont count head 
                # WILL HAVE TO MAKE SURE THE THIS IS IN THE SAME POSITION
                # JUST SWAP POSTIIONS AT THE END 
                elif self.getID() == other_snake.getID():
                    if not other_snakes_found[idx - 2] and curr_pos in other_snake.getPosition()[1:]:
                        look[idx] = distance
                        this_snake_pos = idx
                        other_snakes_found[idx - 2] = True


            # increment the position 
            curr_pos[0] += direction[0]
            curr_pos[1] += direction[1]

            distance += 1

        # wall distance
        look[1] = distance

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

    # apple positions not found 
    def _isCollidingApple(self, position: list, apple_positions: List[list]):
        if position in apple_positions:
            return True
        
        return False
    

    def _isCollidingSnake(self, position: list) -> bool:
        if position in self.snake_position:
            return True
        
        return False


    def __str__(self):

        return str(self.snake_position)