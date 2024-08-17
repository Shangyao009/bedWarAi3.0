import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import pygame
import numpy as np
import sys

from game.TickTimer import TickTimer
from game.Vein import Vein, Bed
from game.Player import Player
from game.structs import Pos, Mine, PlayerId, RewardCollector, Ticks
from game.structs import Direction, ActionId, PlayerObservation
from game.globalConst import Restriction, Reward
import game.Settings as Settings
from game.board import MapBoard, DetailBoard


def create_veins_randomly(
    generator: np.random.Generator, tickTimer: TickTimer
) -> tuple[list[Vein], list[Vein], list[Vein]]:
    created_pos = []
    diamondVeins = []
    goldVeins = []
    ironVeins = []

    def create_pos():
        r: int = generator.integers(0, 8, endpoint=False)
        c: int = generator.integers(0, 8, endpoint=False)
        p = Pos(r, c)
        if (r + c) == 0 or (r + c) == 14:
            return create_pos()
        for pos in created_pos:
            if pos == p:
                return create_pos()
        return p

    def get_symmetric_pos(pos: Pos):
        r = pos.r
        c = pos.c
        if (r + c) == 7:
            return Pos(c, r)
        _r = 7 - r
        _c = c
        _r1 = _c
        _c1 = _r
        r1 = 7 - _r1
        c1 = _c1
        return Pos(r1, c1)

    def create_symmetric_veins(type, num):
        """create num set of veins with symmetric positions"""
        for i in range(num):
            pos = create_pos()
            pos2 = get_symmetric_pos(pos)
            created_pos.append(pos)
            created_pos.append(pos2)
            match (type):
                case Mine.diamond:
                    vein1 = Vein(Mine.diamond, pos, tickTimer)
                    vein2 = Vein(Mine.diamond, pos2, tickTimer)
                    diamondVeins.append(vein1)
                    diamondVeins.append(vein2)
                case Mine.gold:
                    vein1 = Vein(Mine.gold, pos, tickTimer)
                    vein2 = Vein(Mine.gold, pos2, tickTimer)
                    goldVeins.append(vein1)
                    goldVeins.append(vein2)
                case Mine.iron:
                    vein1 = Vein(Mine.iron, pos, tickTimer)
                    vein2 = Vein(Mine.iron, pos2, tickTimer)
                    ironVeins.append(vein1)
                    ironVeins.append(vein2)

    create_symmetric_veins(Mine.diamond, generator.choice([1, 2], p=[0.8, 0.2]))
    create_symmetric_veins(Mine.gold, generator.choice([1, 2], p=(0.8, 0.2)))
    create_symmetric_veins(Mine.iron, generator.choice([1, 2], p=(0.9, 0.1)))

    return (diamondVeins, goldVeins, ironVeins)


class BedWarGame(gym.Env):
    fps = Settings.FPS
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": fps}
    window_height = Settings.WINDOW_HEIGHT
    window_width = Settings.WINDOW_WIDTH

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode: str, human, rgb_array, or None
            handle_pygame_event: callable, a function that handles pygame
        """
        super(BedWarGame, self).__init__()
        self._player_obs_space = spaces.MultiDiscrete(
            [
                *[Restriction.MAX_BLOCK_HEIGHT + 1 for _ in range(64)],
                *[9 for _ in range(24)],
                *[8 for _ in range(8)],
                *[2 for _ in range(3)],
                *[Restriction.MAX_HP + 1 for _ in range(2)],
                Restriction.MAX_HASTE + 1,
                Restriction.MAX_ATK + 1,
                Restriction.MAX_WOOL + 1,
                Restriction.MAX_EMERALD + 1,
                2,
                Restriction.MAX_TRAINING_TIME * 2 + 1,
            ]
        )
        self.observation_space = spaces.Dict(
            {"A": self._player_obs_space, "B": self._player_obs_space}
        )
        self.action_space = spaces.Tuple([spaces.Discrete(20), spaces.Discrete(20)])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # for human render purposes
        self.window = None
        """pygame window"""
        self._font = None
        self._map_board = None
        self._detail_board = None
        self.clock = None

    def _collect_emerald_if_vein_stepped(self):
        """check if player collect emerald from vein, if so, add emerald to player and deduct from vein"""
        A_collect = [0]
        B_collect = [0]

        def collectVein(player: Player, vein: Vein, collect: list[int]):
            if player.is_zero_hp():
                return
            capacity = (
                Restriction.MAX_EMERALD - player.get_emerald_count()
            ) // vein.type.value
            collect_mine = min((capacity, vein.mine_counts))
            collect_mine = 0 if collect_mine < 0 else collect_mine
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
        """return observation for both players"""
        obs_A = self._get_player_obs(PlayerId.Player_A)
        obs_B = self._get_player_obs(PlayerId.Player_B)
        return {"A": obs_A, "B": obs_B}

    def _get_player_obs(self, player_id: PlayerId) -> np.ndarray:
        """return observation for one player"""
        player = self.player_A if player_id == PlayerId.Player_A else self.player_B
        op = self.player_B if player_id == PlayerId.Player_A else self.player_A
        return PlayerObservation(
            height_map=self.height_map,
            diamond_pos=[vein.pos for vein in self.diamond_veins],
            gold_pos=[vein.pos for vein in self.gold_veins],
            iron_pos=[vein.pos for vein in self.iron_veins],
            bed_pos=player.bed.pos,
            op_bed_pos=op.bed.pos,
            pos=player.pos,
            op_pos=op.pos,
            op_alive=op.is_alive(),
            bed_destroyed=player.bed.is_destroyed(),
            op_bed_destroyed=op.bed.is_destroyed(),
            hp=player.hp,
            hp_bound=player.hp_bound,
            haste=player.haste,
            atk=player.atk,
            wool=player.wool,
            emerald=player.emerald,
            in_cd=player.in_cd,
            ticks=self.ticks._val,
        ).to_list()

    def _get_info(self):
        return {}

    def step(
        self, action: tuple[int, int] = [ActionId.NONE, ActionId.NONE]
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], bool, bool, dict]:
        terminated = False
        truncated = False

        if self.game_over:
            return (
                self._get_observation(),
                [0, 0],
                terminated,
                truncated,
                self._get_info(),
            )

        self.reward_collector_A.clear_rewards()
        self.reward_collector_B.clear_rewards()

        self.tick_timer.tick()
        self.ticks += 1

        # make action here
        if not action[0] == ActionId.NONE:
            self.player_A.play_action(action[0], self.player_B)
        if not action[1] == ActionId.NONE:
            self.player_B.play_action(action[1], self.player_A)

        a_collect, b_collect = self._collect_emerald_if_vein_stepped()
        if a_collect > 0:
            self.player_A.reward_collector.add_reward(
                a_collect * Reward.PER_EMERALD_GAIN, f"collect {a_collect} emerald"
            )
        if b_collect > 0:
            self.player_B.reward_collector.add_reward(
                b_collect * Reward.PER_EMERALD_GAIN, f"collect {b_collect} emerald"
            )

        # check if player is dead
        if self.player_A.hp <= 0 and self.player_A.is_alive():
            self.player_A.set_dead_if_zero_hp()
            if not self.player_A.bed.is_destroyed():
                self.player_A.set_revive_timer()
        if self.player_B.hp <= 0 and self.player_B.is_alive():
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
            self.player_A.reward_collector.add_reward(Reward.TIE, "tie")
            self.player_B.reward_collector.add_reward(Reward.TIE, "tie")
            self.game_over = True
        elif A_lose:
            self.player_A.reward_collector.add_reward(Reward.LOSE, "lose")
            self.player_B.reward_collector.add_reward(Reward.WIN, "win")
            self.game_over = True
        elif B_lose:
            self.player_A.reward_collector.add_reward(Reward.WIN, "win")
            self.player_B.reward_collector.add_reward(Reward.LOSE, "lose")
            self.game_over = True

        if not action[0] == ActionId.NONE:
            self.player_A.reward_collector.add_reward(
                Reward.STEP_PENALTY, "step penalty"
            )
            # print("reward A:")
            # for reward_property in self.player_A.reward_collector.rewards:
            #     print(reward_property)
            # print()

        if not action[1] == ActionId.NONE:
            self.player_B.reward_collector.add_reward(
                Reward.STEP_PENALTY, "step penalty"
            )
            # print("reward B:")
            # for reward_property in self.player_B.reward_collector.rewards:
            #     print(reward_property)
            # print()

        reward = (
            self.reward_collector_A.get_total_reward(),
            self.reward_collector_B.get_total_reward(),
        )

        if self.ticks._val >= (Restriction.MAX_TRAINING_TIME * self.fps):
            truncated = True
            # print("max training time exceeded")
            self.game_over = True

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

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

        self.diamond_veins, self.gold_veins, self.iron_veins = create_veins_randomly(
            self.np_random, self.tick_timer
        )

        self.veins: list[Vein] = (
            self.diamond_veins
            + self.gold_veins
            + self.iron_veins
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

        self.game_over = False
        return (self._get_observation(), self._get_info())

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

    def _render_frame(self):
        if self._map_board is None and (
            self.render_mode == "rgb_array" or self.render_mode == "human"
        ):
            pygame.init()
            self._font = pygame.font.Font("freesansbold.ttf", 20)
            self._map_board = MapBoard(self._font)
            self._detail_board = DetailBoard(self._font)

        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.clock = pygame.time.Clock()

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

    def standardize_observation(observation: np.ndarray) -> np.ndarray:
        new_obs = observation.copy().astype(float)
        new_obs[:64] = new_obs[:64] / Restriction.MAX_BLOCK_HEIGHT
        new_obs[64:88] = new_obs[64:88] / 8
        new_obs[88:96] = new_obs[88:96] / 7
        new_obs[99] = new_obs[99] / Restriction.MAX_HP
        new_obs[100] = new_obs[100] / Restriction.MAX_HP
        new_obs[101] = new_obs[101] / Restriction.MAX_HASTE
        new_obs[102] = new_obs[102] / Restriction.MAX_ATK
        new_obs[103] = new_obs[103] / Restriction.MAX_WOOL
        new_obs[104] = new_obs[104] / Restriction.MAX_EMERALD
        new_obs[106] = new_obs[106] / (Restriction.MAX_TRAINING_TIME * 2)
        return new_obs


def parse_key(
    key, heading_A: Direction, heading_B: Direction
) -> tuple[Direction, Direction, ActionId, ActionId]:
    action_A: ActionId = ActionId.NONE
    action_B: ActionId = ActionId.NONE
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
