import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Union
from gymnasium.utils.play import play
import pygame
import numpy as np
import sys

from TickTimer import TickTimer
from Vein import Vein, Bed
from Player import Player
from structs import Pos, Mine, PlayerId, RewardCollector, Ticks, Direction, ActionId
from globalConst import Restriction
import Settings
from board import MapBoard, DetailBoard


class BedWarGame(gym.Env):
    fps = Settings.FPS
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": Settings.FPS}
    window_height = Settings.WINDOW_HEIGHT
    window_width = Settings.WINDOW_WIDTH

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode: str, human, rgb_array, or None
            handle_pygame_event: callable, a function that handles pygame
        """
        super(BedWarGame, self).__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=float
        )
        self.action_space = spaces.Tuple([spaces.Discrete(5), spaces.Discrete(5)])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # for human render purposes
        self.window = None
        """pygame window"""

        self.clock = None
        self.reset()

    def _collect_emerald_if_vein_stepped(self):
        """check if player collect emerald from vein, if so, add emerald to player and deduct from vein"""
        A_collect = [0]
        B_collect = [0]

        def collectVein(player: Player, vein: Vein, collect: list[int]):
            if player.is_zero_hp() == False:
                return
            capacity = (
                Restriction.MAX_EMERALD - player.get_emerald_count()
            ) // vein.type.value
            collect_mine = min((capacity, vein.mine_counts))
            player.collect_emerald(collect_mine * vein.type.value)
            collect[0] = collect_mine * vein.type.value
            vein.mine_counts -= collect_mine

        pos_A = self.player_A.pos
        pos_B = self.player_B.pos
        for vein in self.veins:
            if pos_A != vein.pos and pos_B != vein.pos:
                continue
            if (
                pos_A == vein.pos
                and pos_B == vein.pos
                and self.player_A.is_alive()
                and self.player_B.is_alive()
            ):
                continue
            if pos_A == vein.pos:
                collectVein(self.player_A, vein, A_collect)
            if pos_B == vein.pos:
                collectVein(self.player_B, vein, B_collect)
        return (A_collect[0], B_collect[0])

    def _get_observation(self):
        return None

    def _get_info(self):
        return {}

    def step(
        self, action: Optional[tuple[int, int]] = None
    ) -> tuple[np.ndarray, tuple[int, int], bool, bool, dict]:
        terminated = False
        truncated = False

        self.reward_collector_A.clear_rewards()
        self.reward_collector_B.clear_rewards()

        self.tick_timer.tick()
        self.ticks += 1

        a_collect, b_collect = self._collect_emerald_if_vein_stepped()

        # make action here
        if action is not None:
            if action[0] is not None:
                self.player_A.play_action(action[0], self.player_B)
            if action[1] is not None:
                self.player_B.play_action(action[1], self.player_B)

        # check if player is dead
        if self.player_A.hp <= 0:
            self.player_A.set_dead_if_zero_hp()
            if not self.player_A.bed.is_destroyed():
                self.player_A.set_revive_timer()
        if self.player_B.hp <= 0:
            self.player_B.set_dead_if_zero_hp()
            if not self.player_B.bed.is_destroyed():
                self.player_B.set_revive_timer()

        # check is game over
        A_lose = False
        B_lose = False
        if not self.player_A.is_alive() and self.player_A.bed.is_destroyed():
            A_lose = True
        if not self.player_B.is_alive() and self.player_B.bed.is_destroyed():
            B_lose = True

        if A_lose or B_lose:
            terminated = True
        if A_lose and B_lose:
            print("tie")
            pass
        elif A_lose:
            print("B win")
            pass
        elif B_lose:
            print("A win")
            pass

        reward = [
            self.reward_collector_A.get_total_reward(),
            self.reward_collector_B.get_total_reward(),
        ]
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None):
        # used to seed self.np_random
        super().reset(seed=seed)

        self.ticks = Ticks(0)
        # reset the game
        self.tick_timer = TickTimer(self.fps)
        self.height_map = np.zeros((8, 8)).astype(int)
        self.bed_A = Bed(Pos(0, 0), self.tick_timer)
        self.bed_B = Bed(Pos(7, 7), self.tick_timer)
        self.height_map[self.bed_A.pos.r, self.bed_A.pos.c] = 1
        self.height_map[self.bed_B.pos.r, self.bed_B.pos.c] = 1

        self.diamondVeins = [
            Vein(Mine.diamond, Pos(4, 3), self.tick_timer),
            Vein(Mine.diamond, Pos(3, 4), self.tick_timer),
        ]
        self.goldVeins = [
            Vein(Mine.gold, Pos(1, 5), self.tick_timer),
            Vein(Mine.gold, Pos(6, 2), self.tick_timer),
            Vein(Mine.gold, Pos(5, 7), self.tick_timer),
            Vein(Mine.gold, Pos(2, 0), self.tick_timer),
        ]
        self.ironVeins = [
            Vein(Mine.iron, Pos(5, 5), self.tick_timer),
            Vein(Mine.iron, Pos(2, 2), self.tick_timer),
        ]
        self.veins: list[Vein] = (
            self.diamondVeins
            + self.goldVeins
            + self.ironVeins
            + [self.bed_A, self.bed_B]
        )

        self.reward_collector_A = RewardCollector(ticks=self.ticks)
        self.reward_collector_B = RewardCollector(ticks=self.ticks)
        self.player_A = Player(
            PlayerId.Player_A,
            self.bed_A,
            self.height_map,
            self.tick_timer,
            self.reward_collector_A,
        )
        self.player_B = Player(
            PlayerId.Player_B,
            self.bed_B,
            self.height_map,
            self.tick_timer,
            self.reward_collector_B,
        )
        self.tick_timer.add_timer(
            Settings.DEAD_MATCH_COUNTDOWN,
            self._start_death_match,
            forever=False,
            priority=3,
        )

        return self._get_observation(), self._get_info()

    def _start_death_match(self):
        def deduct_hp():
            self.player_A.injure(1)
            self.player_B.injure(1)

        self.bed_A.set_destroyed()
        self.bed_B.set_destroyed()

        for vein in self.veins:
            vein.stop_produce()

        self.tick_timer.add_timer(
            Settings.DEDUCT_HP_INTERVAL, deduct_hp, forever=True, priority=3
        )

        if not Settings.DEAD_MATCH_TRADE_VALID:
            self.bed_A.set_is_trade_available(False)
            self.bed_B.set_is_trade_available(False)

    def render(self):
        # render the game
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _pygame_init(self):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self._font = pygame.font.Font("freesansbold.ttf", 20)
        self.clock = pygame.time.Clock()

        self._map_board = MapBoard(self._font)
        self._detail_board = DetailBoard(self._font)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self._font = pygame.font.Font("freesansbold.ttf", 20)
            self.clock = pygame.time.Clock()

            self._map_board = MapBoard(self._font)
            self._detail_board = DetailBoard(self._font)

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((10, 10, 10))

        map_canva = self._map_board.update(
            self.height_map,
            self.veins,
            self.player_A.pos,
            self.player_A.heading,
            self.player_B.pos,
            self.player_B.heading,
        )

        detail_canva = self._detail_board.update(
            self.player_A, self.player_B, self.ticks._val
        )

        canvas.blit(map_canva, (70, 70))
        canvas.blit(detail_canva, (670, 70))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def parse_key(key, heading_A: Direction, heading_B: Direction):
    action_A: Optional[ActionId] = None
    action_B: Optional[ActionId] = None
    match key:
        case pygame.K_UP:
            heading_B = Direction.FORWARD
        case pygame.K_DOWN:
            heading_B = Direction.BACK
        case pygame.K_LEFT:
            heading_B = Direction.LEFT
        case pygame.K_RIGHT:
            heading_B = Direction.RIGHT
        case pygame.K_w:
            heading_A = Direction.FORWARD
        case pygame.K_s:
            heading_A = Direction.BACK
        case pygame.K_a:
            heading_A = Direction.LEFT
        case pygame.K_d:
            heading_A = Direction.RIGHT

        case pygame.K_h:
            if heading_A != Direction.CENTER:
                action_A = ActionId.MOVE_F + heading_A
        case pygame.K_j:
            if heading_A != Direction.CENTER:
                action_A = ActionId.PLACE_F + heading_A
        case pygame.K_k:
            if heading_A != Direction.CENTER:
                action_A = ActionId.DIVE_F + heading_A
        case pygame.K_l:
            action_A = ActionId.ATTACK
        case pygame.K_1:
            action_A = ActionId.TRADE_WOOL
        case pygame.K_2:
            action_A = ActionId.HP_POTION
        case pygame.K_3:
            action_A = ActionId.HP_BOUND_UP
        case pygame.K_4:
            action_A = ActionId.HASTE_UP
        case pygame.K_5:
            action_A = ActionId.ATK_UP

        case pygame.K_KP_0:
            if heading_B != Direction.CENTER:
                action_B = ActionId.MOVE_F + heading_B
        case pygame.K_KP_1:
            if heading_B != Direction.CENTER:
                action_B = ActionId.PLACE_F + heading_B
        case pygame.K_KP_2:
            if heading_B != Direction.CENTER:
                action_B = ActionId.DIVE_F + heading_B
        case pygame.K_KP_3:
            action_B = ActionId.ATTACK
        case pygame.K_KP_MINUS:
            action_B = ActionId.TRADE_WOOL
        case pygame.K_KP_PLUS:
            action_B = ActionId.HP_POTION
        case pygame.K_KP_7:
            action_B = ActionId.HP_BOUND_UP
        case pygame.K_KP_8:
            action_B = ActionId.HASTE_UP
        case pygame.K_KP_9:
            action_B = ActionId.ATK_UP

    return (heading_A, heading_B, action_A, action_B)


env = BedWarGame(render_mode="human")
env._pygame_init()
env.reset()
while True:
    env.step()
    if env.render_mode == "human":
        done_action = False  # avoid multiple actions in one frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                key = event.key
                heading_A, heading_B, action_A, action_B = parse_key(
                    key, env.player_A.heading, env.player_B.heading
                )
                env.player_A.heading = heading_A
                env.player_B.heading = heading_B
                if done_action:
                    continue
                if action_A is not None or action_B is not None:
                    env.step((action_A, action_B))
                    done_action = True
        env.render()
    elif env.render_mode == "rgb_array":
        rgb_array = env.render()
