import cv2
import random
import numpy as np
from gym import spaces, Env
from collections import deque

class Actions():
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Snake(Env):

    def __init__(self, size=7, is_human=False, time_between_moves=100, timestep=None):
        super(Snake, self).__init__()

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

        self.size = size
        self.is_human = is_human
        self.time_between_moves = time_between_moves
        self.timestep = timestep
        
        self.max_score = 3
        self.min_moves = (size ** 2 - 1) ** 2
        self.times_won = 0
        # get the whole matrix for the game; used for random apple 
        self.whole_coord = np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist()

        # only need left/right/up because of local direction 
        self.action_space = spaces.Discrete(3)

        obs_size = 8
        directions_size = 24
        prev_action_size = self.size ** 2 + 1 # could also be 100 for health ?? 

        total_size = obs_size + directions_size + prev_action_size
        self.observation_space = spaces.Box(low=-1, high=size**2, shape=(total_size,), dtype=np.int8) # change later 


    def _GetApplePosition(self):
        # making sure the apple doesnt spawn in a snake and another apple 
        choices = [choice for choice in self.whole_coord if choice not in self.snake_positions and choice not in self.apple_positions]
        # make sure the board isn't filled up completely 
        if choices:
            random_choice = random.choice(choices)
            self.apple_positions.append(random_choice)

    
    def SetApplePosition(self, apple_positions):
        self.apple_positions = apple_positions
    
    def SetSnakePosition(self, snake_positions):
        self.snake_positions = snake_positions


    def returnActionFromDirection(self):
        
        if self.direction == 0:
            action = 'left'
        elif self.direction == 1:
            action = 'right'
        elif self.direction == 2:
            action = 'up'
        else:
            action = 'down'

        return action


    def _GetOBS(self):
        # Observation includes:
        # Snake head x, y
        # Snake tail x, y
        # direction 
        # length of snake
        # moves since eaten last apple
        # the 8 directions ; broken up into distance to snake body, apple, and wall 
        #   i think directinos should be calibrated in local instead of global 
        #   check to see if previous way was local/global
        #   think we can simplify down to 4 for loops with negative numbers 
        #   starting with global distances
        # previous actions 

        snake_head = [ self.snake_positions[0][0], self.snake_positions[0][1] ]
        snake_tail = [ self.snake_positions[-1][0], self.snake_positions[-1][1] ]


        # left/right/up/down
        # leftUP/leftDOWN/rightUP/rightDOWN
        directions = [ [-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1] ]

        sight_obs = list()
        for direction in directions:
            sight_obs += self._lookInDirection(direction, snake_head)
        
        obs = [
            snake_head[0],
            snake_head[1],
            snake_tail[0],
            snake_tail[1],
            self.direction,
            self.score,
            self.moves_to_get_apple,
            self.eat_counter
        ] + list(self.prev_actions) \
            + sight_obs

        return np.array(obs)


    # direction should be -1/+1 in tuple (1, -1) for x, y
    def _lookInDirection(self, direction, snake_head):
        
        look = [-1] * 3

        food_found = False
        body_found = False

        distance = 0

        #TODO  we will have to apply the direction before while loop 
        # curr_pos = snake_head
        curr_pos = [snake_head[0] + direction[0], snake_head[1] + direction [1]]

        while not self._isCollidingWall(curr_pos):

            if not food_found and self._isCollidingApple(curr_pos):
                food_found = True
                look[0] = distance
            
            if not body_found and self._isCollidingSnake(curr_pos):
                body_found = True
                look[1] = distance

            # increment the position 
            curr_pos[0] += direction[0]
            curr_pos[1] += direction[1]

            distance += 1

        look[2] = distance

        return look


    def _isCollidingWall(self, position):
        # if we are outside of the walls
        if position[0] >= self.size or \
            position[0] < 0 or \
            position[1] >= self.size or \
            position[1] < 0:

            return True
        
        return False


    def _isCollidingApple(self, position):
        if position in self.apple_positions:
            return True
        
        return False
    

    def _isCollidingSnake(self, position):
        if position in self.snake_positions:
            return True
        
        return False

    # after every step there is a 15% chance to spawn an apple 
    # there is no guarantee that another  apple will spawn if we eat 
    # if there is only 1 food on the board when we eat, we must generate another one 
    def step(self, action):

        info = {
            'won' : False
        }

        snake_head = [ self.snake_positions[0][0], self.snake_positions[0][1] ]

        # if player is not human 
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

        # if player is human we switch to more intuitive controls
        else:

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
        self.total_moves += 1

        # add action to our previous actions
        self.prev_actions.append(action)

        # if we eat an apple 
        if snake_head in self.apple_positions:
            
            self.score += 1
            # we can remove the apple from the list 
            self.apple_positions.remove(snake_head)
            # elongate the snake
            self.snake_positions.insert(0, snake_head)


            # if we eat more than 2 apples, we dont want to pop 
            if self.eat_counter == 0:
                self.snake_positions.pop()

            self.eat_counter = 1

            if self.score > self.max_score:
                self.max_score = self.score

            # if we won the game 
            if self.score >= self.size ** 2 + 1:

                self.times_won += 1
                
                if self.total_moves < self.min_moves:
                    self.min_moves = self.total_moves

                reward = 5
                self.done = True
                info['won'] = True
            
            else:
                reward = (1 - (self.moves_to_get_apple / (self.size ** 2 - 1))) * .5

            # reset counter for next apple
            self.moves_to_get_apple = 0
        
        # else we can just move the snake to the next position
        else:
            
            self.snake_positions.insert(0, snake_head)
            # wait a move to elongate the snake
            if self.eat_counter > 0:
                self.eat_counter = 0
            else:
                self.snake_positions.pop()

            # reward based on if we see an apple? ; not sure if this would work 
            reward = .00001

        # if we ate all the apples, get another one 
        if len(self.apple_positions) == 0:
            self._GetApplePosition()

        # there is a 15% chance to spawn an apple 
        elif random.random() <= .15:
            self._GetApplePosition()

        # checking if the snake collides with itself
        # dont check if the snake collides with itself on the first 3 moves bc we get a list of len 3 of same coordinates ; game :(
        if self.total_moves > 3 and snake_head in self.snake_positions[1:]:
                self.done = True
                reward = -1

        # checking to see if snake collides with walls 
        elif snake_head[0] >= self.size or \
            snake_head[0] < 0 or \
            snake_head[1] >= self.size or \
            snake_head[1] < 0:

            self.done = True
            reward = -1
        
        # if the snake takes too long to get an apple, terminate
        elif self.moves_to_get_apple >= self.size ** 2:
            self.done = True
            reward = -0.5


        # can use the direction for global action to send to server
        info['direction'] = self.direction

        return self._GetOBS(), reward, self.done, info


    def reset(self):
        self.snake_positions = [ [random.randint(0, self.size - 1), random.randint(0, self.size - 1)] ] * 3
        
        self.apple_positions = list()
        # need to call the apple twice bc thats how it is in the game 
        self._GetApplePosition()
        self._GetApplePosition()
        
        self.prev_actions = deque([-1] * (self.size ** 2 + 1), maxlen=(self.size ** 2 + 1))
        
        self.done = False

        self.moves_to_get_apple = 0
        self.total_moves = 0

        # battlesnake health ; aka max moves to get an apple before we die 
        self.health = 100

        # snake is facing up
        self.direction = 2

        self.eat_counter = 0
        self.score = len(self.snake_positions)

        return self._GetOBS()


    def _GetRenderImg(self, renderer=100):
        img = np.zeros((self.size* renderer, self.size * renderer, 3), dtype=np.uint8)

        # drawing the snake
        head = self.snake_positions[0]
        cv2.rectangle(img=img, pt1=(head[0] * renderer, (self.size - head[1]) * renderer - renderer), pt2=(head[0] * renderer + renderer, (self.size - head[1]) * renderer), color=(255,0,0), thickness=-1)
       
        for position in self.snake_positions[1:-1]:
            cv2.rectangle(img=img, pt1=(position[0] * renderer, (self.size - position[1]) * renderer - renderer), pt2=(position[0] * renderer + renderer, (self.size - position[1]) * renderer), color=(0,255,0), thickness=-1)

        if len(self.snake_positions) > 1:
            tail = self.snake_positions[-1]
            cv2.rectangle(img=img, pt1=(tail[0] * renderer, (self.size - tail[1]) * renderer - renderer), pt2=(tail[0] * renderer + renderer, (self.size - tail[1]) * renderer), color=(255,255,255), thickness=-1)

        # drawing the apple
        for apple_position in self.apple_positions:
            cv2.rectangle(img=img, pt1=(apple_position[0] * renderer, (self.size - apple_position[1]) * renderer - renderer), pt2=(apple_position[0] * renderer + renderer, (self.size - apple_position[1]) * renderer), color=(0,0,255), thickness=-1)
       
        # for the STREAM 
        padding = np.full((self.size * renderer, 300, 3), 125, dtype=np.uint8)
        img = np.append(img, padding, axis=1)

        cv2.putText(
            img=img, 
            text=f'Length: {self.score}', 
            org=( ((self.size) * renderer), renderer * 1), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Total Moves: {self.total_moves}', 
            org=( ((self.size) * renderer), renderer * 2), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Min Moves: {self.min_moves}', 
            org=( ((self.size) * renderer), renderer * 3), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Best len: {self.max_score}', 
            org=( ((self.size) * renderer), renderer * 4), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Board size: {self.size}x{self.size}', 
            org=( ((self.size) * renderer), renderer * 5), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        cv2.putText(
            img=img, 
            text=f'Times won: {self.times_won}', 
            org=( ((self.size) * renderer), renderer * 6), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=.8, 
            color=(255, 255, 255), 
            thickness=2
        )

        return img


    def render(self, renderer=100):
        img = self._GetRenderImg(renderer=renderer)
        if self.is_human:
            cv2.imshow('Player', img)
        elif self.timestep:
            cv2.imshow(f'Snake AI {self.timestep}', img)
        else:
            cv2.imshow('Snake AI', img)
        
        cv2.waitKey(self.time_between_moves)
