from gymnasium.envs.registration import register
import gymnasium as gym
from game.BedWarGame import BedWarGame, parse_key, ActionId
import pygame
import sys
import random


def register_game():
    register(
        id="BedWarGame-v0",
        entry_point="game.BedWarGame:BedWarGame",
        additional_wrappers=[],
    )


def play(render_mode=None, seed=None):
    env = gym.make("BedWarGame-v0", render_mode=render_mode)
    observation, info = env.reset(seed=seed)
    env.render()
    while True:
        # action_A = random.choice([i for i in range(19)])
        # action_B = random.choice([i for i in range(19)])
        action_A = ActionId.NONE
        action_B = ActionId.NONE
        env.step([action_A, action_B])
        if env.render_mode == "human":
            done_action = False  # avoid multiple actions in one frame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_r:
                        env.reset(seed=seed)
                        break
                    heading_A, heading_B, action_A, action_B = parse_key(
                        key, env.player_A.heading, env.player_B.heading
                    )
                    env.player_A.heading = heading_A
                    env.player_B.heading = heading_B
                    if done_action:
                        continue
                    if not action_A == ActionId.NONE or not action_B == ActionId.NONE:
                        env.step((action_A, action_B))
                        done_action = True
            env.render()
        elif env.render_mode == "rgb_array":
            rgb_array = env.render()


if __name__ == "__main__":
    register_game()
    # gym.make("BedWarGame-v0", render_mode=None)
    # envs = gym.vector.AsyncVectorEnv(
    #     [make_env for i in range(2)]
    # )
    play(render_mode="human", seed=None)
