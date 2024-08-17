"""all of the common used classes, structs and enum used in the game"""

from enum import Enum
from typing import Optional
import numpy as np
import game.Settings as Settings


class Ticks:
    """Tick obj for sharing ticks between reward collector and game simulator"""

    def __init__(self, val) -> None:
        self._val = int(val)

    def __add__(self, val):
        if isinstance(val, Ticks):
            return Ticks(self._val + val._val)
        return self._val + val

    def __iadd__(self, val):
        self._val += val
        return self


class RewardCollector:
    class RewardProperty:
        def __init__(self, reward: float, info: Optional[str] = None):
            self.reward = reward
            self.info = info

        def __str__(self) -> str:
            return f"reward: {self.reward}, info: {self.info}"

    def __init__(self, ticks: Ticks) -> None:
        self.rewards: list[RewardCollector.RewardProperty] = []
        self.ticks = ticks

    def add_reward(self, reward: float, info: Optional[str] = None):
        self.rewards.append(RewardCollector.RewardProperty(reward, info))

    def get_total_reward(self) -> float:
        total = 0
        for item in self.rewards:
            reward = item.reward
            total += reward
        return total

    def get_rewards_list(self):
        return self.rewards

    def clear_rewards(self):
        self.rewards.clear()


class Pos:
    def __init__(self, r, c) -> None:
        self.c = int(c)
        self.r = int(r)

    def __eq__(self, __value: object) -> bool:
        return self.c == __value.c and self.r == __value.r

    def is_adjacent(self, __value: object) -> bool:
        return abs(self.c - __value.c) <= 1 and abs(self.r - __value.r) <= 1

    def distance(self, __value: object) -> int:
        return abs(self.c - __value.c) + abs(self.r - __value.r)

    def __str__(self) -> str:
        return f"({self.c}, {self.r})"

    def is_invalid(self):
        self.r < 0 or self.r >= 8 or self.c < 0 or self.c >= 8

    # row first then col
    def tolist(self):
        return [self.r, self.c]


class Mine(Enum):
    iron = 1
    gold = 4
    diamond = 16


class Direction:
    CENTER = -1
    FORWARD = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3


class TradeId(Enum):
    wool = 1
    life_potion = 2
    hp_limit_up = 3
    atk_up = 4
    haste_up = 5


class PlayerId:
    Player_A = "Player_A"
    Player_B = "Player_B"


class ActionId:
    DO_NOTHING = 0

    MOVE_F = 1
    MOVE_B = 2
    MOVE_L = 3
    MOVE_R = 4

    DIVE_F = 5
    DIVE_B = 6
    DIVE_L = 7
    DIVE_R = 8

    PLACE_F = 9
    PLACE_B = 10
    PLACE_L = 11
    PLACE_R = 12

    ATTACK = 13

    TRADE_WOOL = 14
    HP_POTION = 15
    HP_BOUND_UP = 16
    HASTE_UP = 17
    ATK_UP = 18

    NONE = 19


class PlayerObservation:
    def __init__(
        self,
        height_map: Optional[np.ndarray] = None,
        diamond_pos: Optional[list[Pos]] = None,
        gold_pos: Optional[list[Pos]] = None,
        iron_pos: Optional[list[Pos]] = None,
        bed_pos: Optional[Pos] = None,
        op_bed_pos: Optional[Pos] = None,
        pos: Optional[Pos] = None,
        op_pos: Optional[Pos] = None,
        op_alive: Optional[bool] = None,
        bed_destroyed: Optional[bool] = None,
        op_bed_destroyed: Optional[bool] = None,
        hp: Optional[int] = None,
        hp_bound: Optional[int] = None,
        haste: Optional[int] = None,
        atk: Optional[int] = None,
        wool: Optional[int] = None,
        emerald: Optional[int] = None,
        in_cd: Optional[bool] = None,
        ticks: Optional[int] = None,
    ) -> None:
        self.height_map = height_map
        self.diamond_pos = diamond_pos
        self.gold_pos = gold_pos
        self.iron_pos = iron_pos
        self.bed_pos = bed_pos
        self.op_bed_pos = op_bed_pos
        self.pos = pos
        self.op_pos = op_pos
        self.op_alive = op_alive
        self.bed_destroyed = bed_destroyed
        self.op_bed_destroyed = op_bed_destroyed
        self.hp = hp
        self.hp_bound = hp_bound
        self.haste = haste
        self.atk = atk
        self.wool = wool
        self.emerald = emerald
        self.in_cd = in_cd
        self.half_second = (2 * ticks) // Settings.FPS

    def veins_to_list(veins: list[Pos], max_len: int):
        """veins amount is within range of 2 to 4, so need to standardize the length"""
        rt = []
        for i in range(max_len):
            if i < len(veins):
                rt.extend(veins[i].tolist())
            else:
                rt.extend([8, 8])
        return rt

    def to_list(self) -> np.ndarray:
        """total len: 107"""
        return np.array(
            [
                *self.height_map.flatten(),  # 64
                *PlayerObservation.veins_to_list(self.diamond_pos, 4),  # 8
                *PlayerObservation.veins_to_list(self.gold_pos, 4),  # 8
                *PlayerObservation.veins_to_list(self.iron_pos, 4),  # 8
                *self.bed_pos.tolist(),  # 2
                *self.op_bed_pos.tolist(),  # 2
                *self.pos.tolist(),  # 2
                *self.op_pos.tolist(),  # 2
                int(self.op_alive),  # 1
                int(self.bed_destroyed),  # 1
                int(self.op_bed_destroyed),  # 1
                self.hp,  # 1
                self.hp_bound,  # 1
                self.haste,  # 1
                self.atk,  # 1
                self.wool,  # 1
                self.emerald,  # 1
                int(self.in_cd),  # 1
                self.half_second,  # 1
            ],
            dtype=np.int64,
        )
