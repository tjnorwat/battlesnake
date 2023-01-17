import cv2
import random
import numpy as np
from typing import List
from gym import spaces, Env
from PlayerData import PlayerData as Player

class Snake(Env):

    def __init__(self, size=7, is_human=False, time_between_moves=100, timestep=None, num_players=1):
        super(Snake, self).__init__()

        # starting positions for the snakes 
        self.starting_positions = [
            [[1, 1], [5, 1]], # bottom left, bottom right
            [[1, 1], [5, 5]], # bottom left, top right
            [[1, 3], [5, 3]], # mid left, mid right
            [[1, 1], [1, 5]], # bottom left, top left
            [[3, 1], [5, 3]], # mid bottom, mid right
            [[1, 5], [5, 5]], # bottom right, top right
            [[3, 5], [5, 3]], # mid top, right mid 
            [[1, 3], [5, 3]] # mid mid, mid mid
        ]

        # BGR !!!!
        self.colors = {
            'red': [[122, 122, 255], [98, 98, 204], [73, 73, 153]],
            'aqua': [[223, 255, 122], [176, 201, 96], [134, 153, 73]],
            'mint': [[159, 255, 122], [128, 204, 98], [96, 153, 73]],
            'cobalt': [[255, 122, 125], [204, 98, 99], [153, 73, 75]],
            'orange': [[122, 178, 255], [98, 142, 204], [73, 106, 153]],
            'yellow': [[122, 217, 255], [98, 174, 204], [73, 131, 153]],
            'lime': [[122, 255, 212], [98, 204, 170], [73, 153, 128]],
            'blue': [[255, 191, 122], [204, 153, 98], [153, 114, 73]], 
            'purple': [[255, 122, 205], [204, 98, 163], [153, 73, 122]],
            'pink': [[226, 122, 255], [181, 98, 204], [136, 73, 153]], 
            'soulless': [[204, 204, 204], [153, 153, 153], [102, 102, 102], [51, 51, 51]]
            }

        self.size = size
        self.is_human = is_human
        self.time_between_moves = time_between_moves
        self.timestep = timestep
        self.num_players = num_players
        
        self.total_moves = 0
        # get the whole matrix for the game; used for random apple 
        self.whole_coord = np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist()

        # only need left/right/up because of local direction 
        self.action_space = spaces.Discrete(3)
        
        shape = self.getOBSShape()
        self.observation_space = spaces.Box(low=-1, high=1, shape=shape, dtype=np.float16) # change later 


    # after every step there is a 15% chance to spawn an apple 
    # there is no guarantee that another  apple will spawn if we eat 
    # if there is only 1 food on the board when we eat, we must generate another one 
    def step(self, actions: np.ndarray):
        
        self.total_moves += 1

        obs = list()
        rewards = list()
        apples_to_delete = list()

        for action, player in zip(actions, self.snake_players):
            reward, apple_to_delete = player.MoveSnake(action, self.apple_positions)
            if apple_to_delete:
                apples_to_delete.append(apple_to_delete)

            rewards.append(reward)

        # have to do it like this :(
        apples_to_delete = set(tuple(i) for i in apples_to_delete)
        # get new apple logic / delete apple 
        for apple in apples_to_delete:
            self.apple_positions.remove(list(apple))
        
        # if we ate all the apples, get another one 
        if len(self.apple_positions) == 0:
            self._GetRandomApplePosition()

        # there is a 15% chance to spawn an apple 
        elif random.random() <= .15:
            self._GetRandomApplePosition()


        # colliding with other snakes 
        # sometimes we change the reward based on if we eat / collide with another snake
        for idx1, player1 in enumerate(self.snake_players):
            for idx2, player2 in enumerate(self.snake_players):
                # make sure the snake isnt dead
                if not player1.isDone():
                    if player1.getID() != player2.getID() and not player2.isDone():
                        
                        # # not sure if this works with more than 2 snakes
                        # # checking to see if both snakes ate an apple 
                        # if (0 < rewards[idx1] <= 1) and (0 < rewards[idx2] <= 1):
                        #     pass
                        # # zero sum reward
                        # elif 0 < rewards[idx1] <= 1:
                        #     rewards[idx2] = -rewards[idx1]


                        snake_collision, collided_with_head = player1.collideWithOtherSnakes(player2)
                        # if we collide with another snake and its a head collision 
                        if snake_collision and collided_with_head:
                            # if the snake is bigger, it wins 
                            player1_score = player1.getScore()
                            player2_score = player2.getScore()
                            if player1_score > player2_score:
                                rewards[idx1] = 3 # reward for "eating" the other snake
                                player2.setDone(True)
                                rewards[idx2] = -3
                            # same length, both players would lose 
                            # this is also the only one that matters in 1v1
                            elif player1_score == player2_score:
                                rewards[idx1] = -3
                                rewards[idx2] = -3
                                player1.setDone(True)
                                player2.setDone(True)
                            
                            elif player1_score < player2_score:
                                rewards[idx1] = -3
                                player1.setDone(True)
                                rewards[idx2] = 3 
                        # otherwise just a collision with another snake body
                        elif snake_collision:
                            rewards[idx1] = -3
                            player1.setDone(True)


        obs = self._GetOBS()

        # need to calculate terminal states
        # check if all snakes are dead 
        done = False

        snake_dones = [player.isDone() for player in self.snake_players]
        # if all snakes are dead [True, True, True] done is true
        if all(snake_dones):
            done = True

        # check if we had more than 2 snakes and 1 is alive / rest are dead
        elif self.num_players >= 2:
            
            # num of alive snakes
            alive_snakes = snake_dones.count(False)
            # dead_snakes = snake_dones.count(True)

            if alive_snakes == 1:
                # make sure we reward the snake for winning 
                for idx, player in enumerate(self.snake_players):
                    if not player.isDone():
                        rewards = [-3] * self.num_players
                        rewards[idx] = 3
                done = True
        
        info = dict()
        return obs, rewards, done, info


    def reset(self):

        # randomly getting starting snake(s)
        self.snake_players = list()
        self._GetRandomSnakePositions()

        self.apple_positions = list()

        # starting with 3 apples in the game
        # actual battlesnake starts with one in the middle and one next to each snake (1v1)
        for _ in range(3):
            self._GetRandomApplePosition()

        return self._GetOBS()


    def getOBSShape(self):
        return np.shape(self.reset()[0])


    def _GetOBS(self):

        # cache all player obs and then create list
        all_player_obs = dict()
        for player in self.snake_players:
            all_player_obs[player.getID()] = player.getOBS(self.snake_players, self.apple_positions)


        # adding the other snakes obs to each snake
        obs = list()
        for id1 in all_player_obs.keys():
            temp_obs = all_player_obs[id1].copy()

            for id2 in all_player_obs.keys():
                if id1 != id2:
                    temp_obs += all_player_obs[id2]

            obs.append(temp_obs)

        return obs


    def _GetRandomApplePosition(self):
        # making sure the apple doesnt spawn in a snake and another apple 

        all_snake_pos = list()
        for player in self.snake_players:
            all_snake_pos += player.getPosition()
        choices = [choice for choice in self.whole_coord if choice not in all_snake_pos and choice not in self.apple_positions]
        # make sure the board isn't filled up completely 
        if choices:
            random_choice = random.choice(choices)
            self.apple_positions.append(random_choice)


    def _GetRandomSnakePositions(self):
        # easy to see what spots have been taken already
        # taken_positions = list()
        random_start_pos = random.choice(self.starting_positions)
        # loop through all players and set their positiion randomly
        for i in range(self.num_players):

            if self.is_human:
                new_player = Player(ID=i, is_human=True, size=self.size)
            else:
                new_player = Player(ID=i, size=self.size) 

            new_player.SetPosition([random_start_pos[i]] * 3)
            self.snake_players.append(new_player)


    # this section is for communicating with the battlesnake API 
    # __________________________________________________________

    def getActionFromDirection(self, player_idx):
        player = self.snake_players[player_idx]
        if player.direction == 0:
            action = 'left'
        elif player.direction == 1:
            action = 'right'
        elif player.direction == 2:
            action = 'up'
        else:
            action = 'down'

        return action

    def setApplePosition(self, apple_positions):
        self.apple_positions = apple_positions

    # figure out how to change these 
    def setSnakePosition(self, snake_positions):
        for player, position in zip(self.snake_players, snake_positions):
            player.SetPosition(position)

    def setLength(self, snake_lengths):
        for player, length in zip(self.snake_players, snake_lengths):
            player.setScore(length)

    def setHealth(self, snake_healths: List[int]):
        for player, health in zip(self.snake_players, snake_healths):
            player.setHealth(health)

    def addActionToList(self, player_idx: int, action: int):
        self.snake_players[player_idx].addToPrevAction(action)

    def addDirection(self, player_idx:int, direction: int):
        self.snake_players[player_idx].setDirection(direction)

    def getPlayerDirection(self, player_idx: int):
        return self.snake_players[player_idx].getDirection()

    # using this to add to prev actions as well as setting direction 
    def setPlayer(self, player_idx: int, action):
        self.snake_players[player_idx].MoveSnake(action, self.apple_positions)

    def getPlayerHead(self, player_idx: int):
        return self.snake_players[player_idx].getHead()

    def stepPlayer(self, player_idx: int, action: int):
        self.snake_players[player_idx].MoveSnake(action=action, apple_positions=self.apple_positions)

    def getApplePositions(self):
        return self.apple_positions

    def ateApple(self, player_idx: int, ate_apple: int):
        self.snake_players[player_idx].setAteApple(ate_apple)

    def getAteApple(self, player_idx: int):
        return self.snake_players[player_idx].getAteApple()

    def getScore(self, player_idx: int):
        return self.snake_players[player_idx].getScore()

    def setBiggest(self, player_idx: int, is_biggest: int):
        self.snake_players[player_idx].setBiggestSnake(is_biggest)

    def getBiggest(self, player_idx: int):
        return self.snake_players[player_idx].getBiggestSnake()


    # ________________________________________________________


    def _GetRenderImg(self, renderer=100):
        img = np.zeros((self.size* renderer, self.size * renderer, 3), dtype=np.uint8)

        colors_list = list(self.colors.values())[1:]
       
        # for the STREAM 
        padding = np.full((self.size * renderer, 300, 3), self.colors['soulless'][2], dtype=np.uint8)
        img = np.append(img, padding, axis=1)

        i = 1
        for color, player in zip(colors_list, self.snake_players):
            head = player.getHead()
            cv2.rectangle(img=img, pt1=(head[0] * renderer, (self.size - head[1]) * renderer - renderer), pt2=(head[0] * renderer + renderer, (self.size - head[1]) * renderer), color=color[0], thickness=-1)

            for position in player.getPosition()[1:-1]:
                cv2.rectangle(img=img, pt1=(position[0] * renderer, (self.size - position[1]) * renderer - renderer), pt2=(position[0] * renderer + renderer, (self.size - position[1]) * renderer), color=color[1], thickness=-1)
            
            tail = player.getTail()
            cv2.rectangle(img=img, pt1=(tail[0] * renderer, (self.size - tail[1]) * renderer - renderer), pt2=(tail[0] * renderer + renderer, (self.size - tail[1]) * renderer), color=color[2], thickness=-1)
            
            if not player.isDone():
                text = f'Agent {i} Len: {player.getScore()} {player.getHealth()}'
            else:
                text = f'Agent {i} Len: {player.getScore()} {player.getHealth()} died'
                
            cv2.putText(
                img=img, 
                text=text, 
                org=( ((self.size) * renderer), renderer * i), 
                fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                fontScale=.8, 
                color=color[1], 
                thickness=2
            )

            i += 1

        # drawing the apple
        for apple_position in self.apple_positions:
            cv2.rectangle(img=img, pt1=(apple_position[0] * renderer, (self.size - apple_position[1]) * renderer - renderer), pt2=(apple_position[0] * renderer + renderer, (self.size - apple_position[1]) * renderer), color=self.colors['red'][1], thickness=-1)

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
