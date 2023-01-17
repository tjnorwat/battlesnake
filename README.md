# TOC
- [TOC](#toc)
- [Battlesnake AI](#battlesnake-ai)
- [Installation](#installation)
- [Usage](#usage)
- [The Files Explained](#the-files-explained)
  - [PlayerData](#playerdata)
  - [SnakeEnvironment](#snakeenvironment)
  - [Main](#main)
  - [MultiInstance](#multiinstance)
  - [Render](#render)
  - [Training](#training)
  - [Server](#server)
  - [Singleplayer](#singleplayer)
  - [SingleInstance](#singleinstance)
- [Next Project and beyond](#next-project-and-beyond)

# Battlesnake AI 
The main goal of this project is to create a custom gym environment in which an agent learns to play the game snake for [BattleSnake](https://play.battlesnake.com/). I have another repo which is a single player mode, which you can check out [here](https://github.com/tjnorwat/snake_ai), where the agent tries to complete the game of snake in the fewest moves possible. This project aims to create an environment where multiple agents can play against each other, mainly 1v1. This required modifying SB3 to be able to support multiple agents in a single env (more about that later). In the end, the agent does OK against others, but ultimately does not perform as well as I would like it due to not "seeing" different types of strategies. 

# Installation
If you would like to install this project and train the snake yourself make sure to install stable baselines and opencv. `pip install requirements.txt` I also used torch and cuda, but you will have to install that on your own if you would like to use those. 

# Usage

In order to train the snakes, you can either use `sb3_multi_train.py` or `sb3_train.py`. The multi training creates multiple environments for the agent to play and is generally faster than only using 1. 

# The Files Explained

This section will try to explain each file more in depth. I could have abstracted the files out more but this was really a proof of concept. The next project will (hopefully) look better :D

## PlayerData

PlayerData stores the data for each snake. This can include 
* position
* direction
* length
* health

and some others. It also is responsible for moving the snake in a direction. An important part about this file is the observation of each snake. In each observation, there is:
* head
* tail
* direction
* if_done
* score / length
* health
* just ate an apple 
* number of apples on the board
* number of alive snakes
* if it is the biggest snake
* sight 
  * consists of 8 different cardinal direction which includes distances from
  * apple 
  * wall
  * itself
  * another snakes * how many snakes there are
  * if we are looking at a snake's head/tail/body
* previous actions

All of this data is normalized between -1 and 1 and then combined with all the other snakes' observations for a complete observation. One neat trick that I started using was only having 3 actions available. Instead of using global direction like left, right, up, down, I am using local direction. This makes it so the agent doesn't have to learn to not go backwards against itself and reduces the action space. It looks a little finicky, but it works. 

## SnakeEnvironment

SnakeEnvironment the custom gym environment that allows SB3 to interact with the game. I tried to recreate the battlesnake environment as much as possible in python. My hope is that I would be able to interact directly with the Battlesnake source, but that is for the next version. 

## Main

This is one of the files that battlesnake provides to interact with their game. I made some modifications for it to also work with my custom environment. Basically just manually stepping each snake and making sure that I am getting the direction for the other snake since battlesnake operates with global direction. 

## MultiInstance

Overrides the DummyVecEnv class from SB3 to be able to run multiple environments at once. Essentially, we just create a list of how many environments we want to run multiplied by the number of agents that are playing in an environment. Since this is for 2 players (1v1) and let's say we would like to have 4 environments running at the same time, we create 8 VecEnvs. This acts as 2 agents that are playing in a single game and we slice the actions that we get from the model to step through each env with multiple agents. 

## Render

Renders a simplistic visualization of the game and automatically gets the current version of the model after a certain period of time. Useful for streaming and seeing progress throughout the training period. 

## Training 

Both the training files are simplistic and just train the agents. I didn't explore too much with hyperparameters but did end up changing the network architecture. Creating a separate network for the policy and value function seemed to help. 

## Server

Battlesnake provided to interact with their API.

## Singleplayer

Allows you to interact with the custom gym environment. Doesn't support going against the agent, more of a testing environment to easily see observation and rewards.

## SingleInstance

Overrides the VecEnv class to train on a single environment. This was more a stepping stone to help make the MultiInstance class. You can use this to train, but it is a lot slower since only 1 environment is used. 

# Next Project and beyond 

The next version of this project will hopefully:
* interact with the source of battlesnake 
* be able to train on multiple computers 
* explore and perfect different types of strategies
* become the best snake 