import cv2
import random
import numpy as np
from gym import spaces, Env
from PlayerData import PlayerData as Player

class Snake(Env):

    def __init__(self, size=7, is_human=False, time_between_moves=100, timestep=None, num_players=1):
        # super(Snake, self).__init__()
        super().__init__()

        # BGR !!!!
        self.colors = {
            'red': [[122, 122, 255], [98, 98, 204], [73, 73, 153]],
            'orange': [[122, 178, 255], [98, 142, 204], [73, 106, 153]],
            'yellow': [[122, 217, 255], [98, 174, 204], [73, 131, 153]],
            'lime': [[122, 255, 212], [98, 204, 170], [73, 153, 128]],
            'mint': [[159, 255, 122], [128, 204, 98], [96, 153, 73]],
            'aqua': [[223, 255, 122], [176, 201, 96], [134, 153, 73]],
            'blue': [[255, 191, 122], [204, 153, 98], [153, 114, 73]], 
            'cobalt': [[255, 122, 125], [204, 98, 99], [153, 73, 75]],
            'purple': [[255, 122, 205], [204, 98, 163], [153, 73, 122]],
            'pink': [[226, 122, 255], [181, 98, 204], [136, 73, 153]], 
            'soulless': [[204, 204, 204], [153, 153, 153], [102, 102, 102], [51, 51, 51]]
            }

        self.size = size
        self.is_human = is_human
        self.time_between_moves = time_between_moves
        self.timestep = timestep
        self.num_players = num_players
        
        # get the whole matrix for the game; used for random apple 
        self.whole_coord = np.mgrid[0:size, 0:size].reshape(2, -1).T.tolist()

        # calculating observation shape 
        obs_size = 11
        directions_size = 3 * 8 + (num_players - 1) * 8
        # prev_action_size = self.size ** 2 + 1 # could also be 100 for health ?? 
        prev_action_size = 100  

        total_size = obs_size + directions_size + prev_action_size

        # high value could either be size of board or health 
        if size ** 2 > 100:
            high = size ** 2
        else:
            high = 100

        # only need left/right/up because of local direction 
        self.action_space = spaces.Discrete(3)
        # self.action_space = spaces.MultiDiscrete([3] * num_players)
        
        shape = (num_players, total_size * num_players)
        # shape = (total_size * num_players * num_players,)
        print('final shape', shape)
        self.observation_space = spaces.Box(low=-1, high=high, shape=shape, dtype=np.float16) # change later 


    # after every step there is a 15% chance to spawn an apple 
    # there is no guarantee that another  apple will spawn if we eat 
    # if there is only 1 food on the board when we eat, we must generate another one 
    def step(self, actions: np.ndarray):

        obs = list()
        rewards = list()
        apples_to_delete = list()

        print('ACTIONS', actions)
        # if not isinstance(actions, list):
        #     actions = [actions]
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
        for idx, player1 in enumerate(self.snake_players):
            for player2 in self.snake_players:
                # make sure the snake isnt dead
                if not player1.isDone():
                    if player1.getID() != player2.getID() and not player2.isDone():
                        snake_collision, collided_with_head = player1.collideWithOtherSnakes(player2)
                        # if we collide with another snake and its a head collision 
                        if snake_collision and collided_with_head:
                            # if the snake is bigger, it wins 
                            player1_score = player1.getScore()
                            player2_score = player2.getScore()
                            if player1_score > player2_score:
                                rewards[idx] = 2 # reward for "eating" the other snake
                                
                            # if same length or smaller, both would lose 
                            elif player1_score <= player2_score:
                                rewards[idx] = -1
                                player1.setDone(True)
                        # otherwise just a collision with another snake body
                        elif snake_collision:
                            rewards[idx] = -1
                            player1.setDone(True)

        obs = self._GetOBS()
        # print('obs shape', np.shape(obs))


        # need to calculate terminal states
        # check if all snakes are dead 
        done = False

        # if all snakes are dead [True, True, True] done is true
        if all(player.isDone() is True for player in self.snake_players):
            done = True

        # check if we had more than 2 snakes and 1 is alive / rest are dead
        elif self.num_players >= 2:

            snake_dones = [player.isDone() for player in self.snake_players]
            
            # num of alive snakes
            alive_snakes = snake_dones.count(False)
            # dead_snakes = snake_dones.count(True)

            if alive_snakes == 1:
                # make sure we reward the snake for winning 
                for idx, player in enumerate(self.snake_players):
                    if not player.isDone():
                        rewards[idx] = 5
                done = True

        info = dict()
        
        return np.array(obs), np.array(rewards), done, info


    def reset(self):
        # self.snake_positions = [ [random.randint(0, self.size - 1), random.randint(0, self.size - 1)] ] * 3

        # randomly getting starting snake(s)
        self.snake_players = list()
        self._GetRandomSnakePositions()

        self.apple_positions = list()
        # need to call the apple twice bc thats how it is in the game
        # game always starts with 2 apples for 7x7 
        # 3 for 11x11
        self._GetRandomApplePosition()
        self._GetRandomApplePosition()
        if self.size == 11:
            self._GetRandomApplePosition()

        return np.array(self._GetOBS())


    def _GetOBS(self):
        
        obs = list()

        for player1 in self.snake_players:
            # first we get the player obs and then we can append the other snakes
            # observation includes that of the player 
            # how close it is to other snakes
            # what the observations of other snakes
            # for each player
            temp_obs = player1.getOBS(self.snake_players, self.apple_positions)

            for player2 in self.snake_players:
                # already got player above
                if player1.getID() != player2.getID():
                    # extending current observation of specific player
                    temp_obs.extend(player2.getOBS(self.snake_players, self.apple_positions))

            obs.append(temp_obs)

        print('other shape', np.shape(obs))
        # obs = np.concatenate(obs)
        # return obs
        return np.array(obs)


    def _GetRandomApplePosition(self):
        # making sure the apple doesnt spawn in a snake and another apple 

        all_snake_pos = list()
        for player in self.snake_players:
            all_snake_pos.extend(player.getPosition())

        choices = [choice for choice in self.whole_coord if choice not in all_snake_pos and choice not in self.apple_positions]
        # make sure the board isn't filled up completely 
        if choices:
            random_choice = random.choice(choices)
            self.apple_positions.append(random_choice)


    def _GetRandomSnakePositions(self):
        # easy to see what psots have been taken already
        taken_positions = list()

        # loop through all players and set their positiion randomly
        for i in range(self.num_players):

            # if this is the first snake just choose a random spot
            if not self.snake_players:

                if self.is_human:
                    new_player = Player(ID=i, is_human=True, size=self.size)
                else:
                    new_player = Player(ID=i, size=self.size) 

                random_pos = random.choice(self.whole_coord)
                taken_positions.append(random_pos)
                new_player.SetPosition([random_pos] * 3)
                self.snake_players.append(new_player)

            # make sure we dont occupy the same space as other snakes 
            else:
                # new_player = Player(ID=i, size=self.size) 
                new_player = Player(ID=i, is_human=True, size=self.size) # TESTING PURPOSE

                choices = [choice for choice in self.whole_coord if choice not in taken_positions]
                random_pos = random.choice(choices)
                taken_positions.append(random_pos)
                new_player.SetPosition([random_pos] * 3)
                self.snake_players.append(new_player)


    # this is for communicating with the battlesnake API 
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

    def SetApplePosition(self, apple_positions):
        self.apple_positions = apple_positions
    

    # figure out how to change these 
    def SetSnakePosition(self, snake_positions):
        self.snake_positions = snake_positions


    def _GetRenderImg(self, renderer=100):
        img = np.zeros((self.size* renderer, self.size * renderer, 3), dtype=np.uint8)

        colors_list = list(self.colors.values())[1:]

        for color, player in zip(colors_list, self.snake_players):
            if not player.isDone():
                head = player.getHead()
                cv2.rectangle(img=img, pt1=(head[0] * renderer, (self.size - head[1]) * renderer - renderer), pt2=(head[0] * renderer + renderer, (self.size - head[1]) * renderer), color=color[0], thickness=-1)

                for position in player.getPosition()[1:-1]:
                    cv2.rectangle(img=img, pt1=(position[0] * renderer, (self.size - position[1]) * renderer - renderer), pt2=(position[0] * renderer + renderer, (self.size - position[1]) * renderer), color=color[1], thickness=-1)
                
                tail = player.getTail()
                cv2.rectangle(img=img, pt1=(tail[0] * renderer, (self.size - tail[1]) * renderer - renderer), pt2=(tail[0] * renderer + renderer, (self.size - tail[1]) * renderer), color=color[2], thickness=-1)

        # drawing the apple
        for apple_position in self.apple_positions:
            cv2.rectangle(img=img, pt1=(apple_position[0] * renderer, (self.size - apple_position[1]) * renderer - renderer), pt2=(apple_position[0] * renderer + renderer, (self.size - apple_position[1]) * renderer), color=self.colors['red'][1], thickness=-1)

        # display for each snake 
        # * length 
        # * how many moves it has left/done


        # # drawing the snake
        # head = self.snake_positions[0]
        # cv2.rectangle(img=img, pt1=(head[0] * renderer, (self.size - head[1]) * renderer - renderer), pt2=(head[0] * renderer + renderer, (self.size - head[1]) * renderer), color=(255,0,0), thickness=-1)
       
        # for position in self.snake_positions[1:-1]:
        #     cv2.rectangle(img=img, pt1=(position[0] * renderer, (self.size - position[1]) * renderer - renderer), pt2=(position[0] * renderer + renderer, (self.size - position[1]) * renderer), color=(0,255,0), thickness=-1)

        # if len(self.snake_positions) > 1:
        #     tail = self.snake_positions[-1]
        #     cv2.rectangle(img=img, pt1=(tail[0] * renderer, (self.size - tail[1]) * renderer - renderer), pt2=(tail[0] * renderer + renderer, (self.size - tail[1]) * renderer), color=(255,255,255), thickness=-1)

        # # drawing the apple
        # for apple_position in self.apple_positions:
        #     cv2.rectangle(img=img, pt1=(apple_position[0] * renderer, (self.size - apple_position[1]) * renderer - renderer), pt2=(apple_position[0] * renderer + renderer, (self.size - apple_position[1]) * renderer), color=(0,0,255), thickness=-1)
       
        # # for the STREAM 
        # padding = np.full((self.size * renderer, 300, 3), 125, dtype=np.uint8)
        # img = np.append(img, padding, axis=1)

        # cv2.putText(
        #     img=img, 
        #     text=f'Length: {self.score}', 
        #     org=( ((self.size) * renderer), renderer * 1), 
        #     fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        #     fontScale=.8, 
        #     color=(255, 255, 255), 
        #     thickness=2
        # )

        # cv2.putText(
        #     img=img, 
        #     text=f'Total Moves: {self.total_moves}', 
        #     org=( ((self.size) * renderer), renderer * 2), 
        #     fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        #     fontScale=.8, 
        #     color=(255, 255, 255), 
        #     thickness=2
        # )

        # cv2.putText(
        #     img=img, 
        #     text=f'Min Moves: {self.min_moves}', 
        #     org=( ((self.size) * renderer), renderer * 3), 
        #     fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        #     fontScale=.8, 
        #     color=(255, 255, 255), 
        #     thickness=2
        # )

        # cv2.putText(
        #     img=img, 
        #     text=f'Best len: {self.max_score}', 
        #     org=( ((self.size) * renderer), renderer * 4), 
        #     fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        #     fontScale=.8, 
        #     color=(255, 255, 255), 
        #     thickness=2
        # )

        # cv2.putText(
        #     img=img, 
        #     text=f'Board size: {self.size}x{self.size}', 
        #     org=( ((self.size) * renderer), renderer * 5), 
        #     fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        #     fontScale=.8, 
        #     color=(255, 255, 255), 
        #     thickness=2
        # )

        # cv2.putText(
        #     img=img, 
        #     text=f'Times won: {self.times_won}', 
        #     org=( ((self.size) * renderer), renderer * 6), 
        #     fontFace=cv2.FONT_HERSHEY_DUPLEX, 
        #     fontScale=.8, 
        #     color=(255, 255, 255), 
        #     thickness=2
        # )

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
