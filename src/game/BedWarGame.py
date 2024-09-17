import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from game.TickTimer import TickTimer
from game.Vein import Vein, Bed
from game.Player import Player
from game.structs import Pos, PlayerId, RewardCollector, Ticks, TradeId
from game.structs import ActionId, PlayerObservation, Mine
from game.globalConst import Restriction, Reward, TRADE_COST
import game.Settings as Settings
from game.board import MapBoard, DetailBoard
from game.utils import create_veins_randomly, get_valid_actions_mask


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
                *[Restriction.MAX_BLOCK_HEIGHT + 1 for _ in range(5)],
                *[9 for _ in range(28)],
                *[8 for _ in range(4)],
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

    def _get_observation(
        self, obs_A: PlayerObservation, obs_B: PlayerObservation
    ) -> dict:
        """return observation for both players"""
        return {"A": obs_A.to_list(), "B": obs_B.to_list()}

    def _get_player_obs(self, player_id: PlayerId) -> PlayerObservation:
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
        )

    def _get_info(self, obs_A: PlayerObservation, obs_B: PlayerObservation):
        valid_actions_mask_A = get_valid_actions_mask(obs_A)
        valid_actions_mask_B = get_valid_actions_mask(obs_B)
        return {
            "valid_actions_mask_A": valid_actions_mask_A,
            "valid_actions_mask_B": valid_actions_mask_B,
        }

    def _calc_close_to_vein_weight(self, pos: Pos):
        total_weight = 0
        total_weighted_near = 0
        for vein in self.veins:
            total_weight += vein.type.value
            total_weighted_near += (
                (14 - pos.distance(vein.pos)) * vein.type.value
            ) ** 2
        return (total_weighted_near / total_weight) ** 0.5

    def _get_close_to_veins_reward(self, pos: Pos):
        reward = self.close_to_veins_reward_map[pos.r, pos.c] * Reward.CLOSE_TO_VEINS
        return max(reward, 0.0)

    def step(
        self, action: tuple[int, int] = [ActionId.NONE, ActionId.NONE]
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], bool, bool, dict]:
        terminated = False
        truncated = False

        obs_A = self._get_player_obs(PlayerId.Player_A)
        obs_B = self._get_player_obs(PlayerId.Player_B)
        if self.game_over:
            return (
                self._get_observation(obs_A, obs_B),
                [0, 0],
                terminated,
                truncated,
                self._get_info(obs_A, obs_B),
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

        obs_A = self._get_player_obs(PlayerId.Player_A)
        obs_B = self._get_player_obs(PlayerId.Player_B)
        observation = self._get_observation(obs_A, obs_B)
        info = self._get_info(obs_A, obs_B)

        if not action[0] == ActionId.NONE:
            if self.player_A.is_alive():
                _reward = self._get_close_to_veins_reward(self.player_A.pos)
                self.player_A.reward_collector.add_reward(
                    _reward,
                    "close to vein reward",
                )
                if self.player_A.wool > 0:
                    self.player_A.reward_collector.add_reward(
                        Reward.HOLD_WOOL, "holding wool"
                    )

            self.player_A.reward_collector.add_reward(
                Reward.STEP_PENALTY, "step penalty"
            )

        if not action[1] == ActionId.NONE:
            if self.player_B.is_alive():
                _reward = self._get_close_to_veins_reward(self.player_B.pos)
                self.player_B.reward_collector.add_reward(
                    _reward,
                    "close to vein reward",
                )
                if self.player_B.wool > 0:
                    self.player_B.reward_collector.add_reward(
                        Reward.HOLD_WOOL, "holding wool"
                    )
            self.player_B.reward_collector.add_reward(
                Reward.STEP_PENALTY, "step penalty"
            )

        if not self.game_over and self.ticks._val >= (
            Restriction.MAX_TRAINING_TIME * self.fps
        ):
            if Restriction.IS_DONE_IF_TIME_EXCEED:
                terminated = True
                self.player_A.reward_collector.add_reward(
                    Reward.TIE, "tie as max training time exceeded"
                )
                self.player_B.reward_collector.add_reward(
                    Reward.TIE, "tie as max training time exceeded"
                )
            else:
                truncated = True
            # print("max training time exceeded")
            self.game_over = True

        reward = (
            self.reward_collector_A.get_total_reward(),
            self.reward_collector_B.get_total_reward(),
        )

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

        # self.diamond_veins = [
        #     Vein(Mine.diamond, Pos(4, 3), self.tick_timer),
        #     Vein(Mine.diamond, Pos(3, 4), self.tick_timer),
        # ]
        # self.gold_veins = [
        #     Vein(Mine.gold, Pos(1, 5), self.tick_timer),
        #     Vein(Mine.gold, Pos(6, 2), self.tick_timer),
        #     Vein(Mine.gold, Pos(5, 7), self.tick_timer),
        #     Vein(Mine.gold, Pos(2, 0), self.tick_timer),
        # ]
        # self.iron_veins = [
        #     Vein(Mine.iron, Pos(5, 5), self.tick_timer),
        #     Vein(Mine.iron, Pos(2, 2), self.tick_timer),
        # ]

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
        self.is_death_match = False

        # for close to vein reward calculation
        _reward = np.zeros((8, 8))
        for r in range(8):
            for c in range(8):
                _reward[r, c] = self._calc_close_to_vein_weight(Pos(r, c))
        _min = _reward.min()
        _max = _reward.max()
        self.close_to_veins_reward_map = (_reward - _min) / (_max - _min)

        self.game_over = False
        obs_A = self._get_player_obs(PlayerId.Player_A)
        obs_B = self._get_player_obs(PlayerId.Player_B)
        return (self._get_observation(obs_A, obs_B), self._get_info(obs_A, obs_B))

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
        self.is_death_match = True

        if not Settings.DEAD_MATCH_TRADE_VALID:
            self.bed_A.set_is_trade_available(False)
            self.bed_B.set_is_trade_available(False)

    def render(self):
        # render the game
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def get_ticks(self):
        return self.ticks._val

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
