from gymnasium.envs.registration import register
import gymnasium as gym
from game import BedWarGame
import pygame
import sys
import random


def register_game():
    register(
        id="BedWarGame-v0",
        entry_point="game:BedWarGame",
        additional_wrappers=[],
    )


