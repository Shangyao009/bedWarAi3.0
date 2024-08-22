from gymnasium.envs.registration import register
import gymnasium as gym
from game import BedWarGame, ActionId, parse_key, globalConst
import pygame
import sys
import random
from utils import register_game
import numpy as np


def play(render_mode=None, seed=None, n_episodes=1):
    skip_frames = 4
    env = gym.make("BedWarGame-v0", render_mode=render_mode)
    env = gym.wrappers.AutoResetWrapper(env)
    for epoch in range(n_episodes):
        # done = False
        observation, info = env.reset(seed=seed)
        env.render()
        while True:
            done_action = False  # avoid multiple actions in one frame
            if env.ticks._val % skip_frames == 0:
                valid_actions_mask_A = info["valid_actions_mask_A"]
                eligible_actions_A = np.flatnonzero(valid_actions_mask_A)
                action_A = np.random.choice(eligible_actions_A)
                valid_actions_mask_B = info["valid_actions_mask_B"]
                eligible_actions_B = np.flatnonzero(valid_actions_mask_B)
                action_B = np.random.choice(eligible_actions_B)

                observation, _, terminated, truncated, info = env.step(
                    (action_A, action_B)
                )
                # done = terminated or truncated
                done_action = True
            if env.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        key = event.key
                        if key == pygame.K_r:
                            env.reset(seed=seed)
                            break
                        # heading_A, heading_B, action_A, action_B = parse_key(
                        #     key, env.player_A.heading, env.player_B.heading
                        # )
                        # env.player_A.heading = heading_A
                        # env.player_B.heading = heading_B
                        # if done_action:
                        #     continue
                        # if not action_A == ActionId.NONE or not action_B == ActionId.NONE:
                        #     observation, _, terminated, truncated, info = env.step((action_A, action_B))
                        #     done_action = True
                        #     done = terminated or truncated

            if not done_action:
                env.step((ActionId.NONE, ActionId.NONE))

            env.render()

            # if done:
            #     env.reset(seed=seed)


if __name__ == "__main__":
    register_game()
    play(render_mode=None, seed=None, n_episodes=5)
