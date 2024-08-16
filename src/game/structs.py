"""all of the common used classes, structs and enum used in the game"""

from enum import Enum
from typing import Optional


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

    def __init__(self, ticks: Ticks) -> None:
        self.rewards: list[RewardCollector.RewardProperty] = []
        self.ticks = ticks

    def add_reward(self, reward: float, info: Optional[str] = None):
        self.rewards.append(RewardCollector.RewardProperty(reward, info))

    def get_total_reward(self) -> float:
        total = 0
        for item in self.rewards:
            reward = item[0]
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

    def matchMoveActionId(heading):
        if heading == Direction.FORWARD:
            return ActionId.MOVE_F
        if heading == Direction.BACK:
            return ActionId.MOVE_B
        if heading == Direction.LEFT:
            return ActionId.MOVE_L
        if heading == Direction.RIGHT:
            return ActionId.MOVE_R

        return ActionId.DO_NOTHING

    def matchDiveActionId(heading):
        if heading == Direction.FORWARD:
            return ActionId.DIVE_F
        if heading == Direction.BACK:
            return ActionId.DIVE_B
        if heading == Direction.LEFT:
            return ActionId.DIVE_L
        if heading == Direction.RIGHT:
            return ActionId.DIVE_R

        return ActionId.DO_NOTHING

    def matchPlaceActionId(heading):
        if heading == Direction.FORWARD:
            return ActionId.PlACE_F
        if heading == Direction.BACK:
            return ActionId.PLACE_B
        if heading == Direction.LEFT:
            return ActionId.PLACE_L
        if heading == Direction.RIGHT:
            return ActionId.PLACE_R

        return ActionId.DO_NOTHING
