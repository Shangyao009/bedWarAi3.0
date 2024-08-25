from game.Vein import Bed, Pos
from game.TickTimer import TickTimer
from game.structs import Direction, RewardCollector, TradeId, ActionId
import game.Settings as Settings
from game.globalConst import *


class Player:
    def __init__(
        self,
        id: str,
        bed: Bed,
        height_map,
        tick_timer: TickTimer,
        reward_collector: RewardCollector,
    ):
        self.pos = Pos(bed.pos.r, bed.pos.c)
        self.id = id
        self.hp_bound = 20
        self.hp = 20
        self.alive = True
        self.haste = 0
        self.atk = 1

        self.wool = 8
        self.emerald = 0

        self.bed = bed
        self.reward_collector = reward_collector
        self.heading = Direction.CENTER
        """for human play only"""

        self.height_map = height_map

        self.in_cd = False
        """shared cd for attack and dive"""
        self.is_moving = False
        self.tick_timer = tick_timer

    def is_in_cd(self):
        return self.in_cd

    def calc_cd(self, haste):
        return max(Settings.INIT_CD - haste * Settings.HASTE_DEDUCT_CD, Settings.MIN_CD)

    def is_zero_hp(self):
        return self.hp <= 0

    def set_cd_timer(self):
        def callback():
            self.in_cd = False

        self.in_cd = True
        interval = self.calc_cd(self.haste)
        self.tick_timer.add_timer(interval, callback, forever=False)

    def set_revive_timer(self):
        self.tick_timer.add_timer(
            seconds=Settings.REVIVE_CYCLE, task=self.revive, forever=False
        )

    def is_alive(self):
        # hp == 0 is not dead, but will be dead in next step
        return self.alive

    def set_dead_if_zero_hp(self):
        """
        give death penalty, set player hp to 0, move player to (-1,-1)
        often called ofter both player act (even if hp is 0, player can still act in that step)
        """
        if not self.is_alive():
            return
        if self.hp > 0:
            return
        self.hp = 0
        self.alive = False
        self.reward_collector.add_reward(Reward.DEATH, "death")
        self.pos = Pos(8, 8)

    def parse_direction(self, direction: Direction):
        dr = 0
        dc = 0
        match (direction):
            case Direction.FORWARD:
                dr = -1
            case Direction.BACK:
                dr = 1
            case Direction.LEFT:
                dc = -1
            case Direction.RIGHT:
                dc = 1

        return dr, dc

    def is_invalid_block(self, r, c):
        return r < 0 or r > 7 or c < 0 or c > 7

    def revive(self):
        """recover hp and set position to bed"""
        self.hp = self.hp_bound
        self.alive = True
        self.pos.r = self.bed.pos.r
        self.pos.c = self.bed.pos.c

    def move(self, direction: Direction):
        """move func, delay moving for MOVE_INTERVAL seconds"""
        self.heading = direction

        dr, dc = self.parse_direction(direction)
        target_r = self.pos.r + dr
        target_c = self.pos.c + dc

        if (
            not self.is_alive()
            or self.is_moving
            or self.is_invalid_block(target_r, target_c)
        ):
            self.reward_collector.add_reward(Reward.INVALID_ACTION, "invalid move")
            return

        def callback():
            self.pos.c = target_c
            self.pos.r = target_r

            if self.height_map[target_r][target_c] == 0:
                self.hp = 0
            self.is_moving = False

        if Settings.MOVE_INTERVAL == 0:
            callback()
        else:
            self.is_moving = True
            self.tick_timer.add_timer(
                seconds=Settings.MOVE_INTERVAL, task=callback, forever=False
            )

    def dive(self, direction: Direction, op: "Player"):
        invalid_action = False
        if not self.is_alive():
            invalid_action = True
        elif self.in_cd:
            invalid_action = True

        dr, dc = self.parse_direction(direction)
        target_r = self.pos.r + dr
        target_c = self.pos.c + dc

        if (
            self.is_invalid_block(target_r, target_c)
            or self.height_map[target_r][target_c] <= 0
        ):
            invalid_action = True

        if Pos(target_r, target_c) == op.pos and op.is_alive():
            invalid_action = True

        if invalid_action:
            self.reward_collector.add_reward(Reward.INVALID_ACTION, "invalid dive")
            return

        self.height_map[target_r][target_c] -= 1
        if Pos(target_r, target_c) == self.bed.pos and not self.bed.is_destroyed():
            self.reward_collector.add_reward(Reward.ATTACK_SELF_BED, "attack self bed")
            if self.height_map[target_r][target_c] == 0:
                self.bed.set_destroyed()
                self.reward_collector.add_reward(
                    Reward.BED_DESTROYED, "destroy self bed"
                )
        if Pos(target_r, target_c) == op.bed.pos and not op.bed.is_destroyed():
            self.reward_collector.add_reward(Reward.ATTACK_BED, "attack op bed")
            if self.height_map[target_r][target_c] == 0:
                self.reward_collector.add_reward(Reward.DESTROY_BED, "destroy op bed")
                op.bed.set_destroyed()
                op.reward_collector.add_reward(Reward.BED_DESTROYED, "bed destroyed")

        self.set_cd_timer()

    def place_block(self, direction: Direction, op: "Player"):
        invalid_action = False
        if not self.is_alive():
            invalid_action = True
        elif self.wool <= 0:
            invalid_action = True

        dr, dc = self.parse_direction(direction)
        target_r = self.pos.r + dr
        target_c = self.pos.c + dc
        if (
            self.is_invalid_block(target_r, target_c)
            or self.height_map[target_r][target_c] >= Restriction.MAX_BLOCK_HEIGHT
        ):
            invalid_action = True

        if invalid_action:
            self.reward_collector.add_reward(
                Reward.INVALID_ACTION, "invalid place block"
            )
            return

        self.height_map[target_r][target_c] += 1
        self.wool -= 1
        if Pos(target_r, target_c) == self.bed.pos and not self.bed.is_destroyed():
            disc_reward = (
                Reward.CONSTRUCT_BED * (16 - self.height_map[target_r][target_c]) / 16
            )
            disc_reward = max(0, disc_reward)
            self.reward_collector.add_reward(disc_reward, "construct bed")
        if Pos(target_r, target_c) == op.bed.pos and not op.bed.is_destroyed():
            self.reward_collector.add_reward(-Reward.ATTACK_BED, "construct op bed")

    def injure(self, damage):
        self.hp = max(0, self.hp - damage)
        damage = min(self.hp, damage)
        reward = damage * Reward.PER_DAMAGE_TAKE
        self.reward_collector.add_reward(reward, "injured")
        return damage

    def attack(self, op: "Player"):
        invalid_action = False
        if not self.is_alive():
            invalid_action = True
        elif self.in_cd:
            invalid_action = True
        elif not op.is_alive():
            invalid_action = True
        elif not self.pos.is_adjacent(op.pos):
            invalid_action = True

        if invalid_action:
            self.reward_collector.add_reward(Reward.INVALID_ACTION, "invalid attack")
            return

        damage_deal = op.injure(self.atk)
        self.reward_collector.add_reward(Reward.PER_DAMAGE_DEAL * damage_deal, "attack")
        if not op.is_alive():
            self.reward_collector.add_reward(Reward.KILL_OP, "kill op")

        self.set_cd_timer()

    def trade(self, trade_id: TradeId):
        invalid_action = False
        if not self.is_alive():
            invalid_action = True
        elif self.pos != self.bed.pos:
            invalid_action = True
        elif not self.bed.is_trade_available():
            invalid_action = True
        elif TRADE_COST[trade_id] > self.emerald:
            invalid_action = True
        elif TRADE_COST.get(trade_id) is None:
            invalid_action = True

        if invalid_action:
            self.reward_collector.add_reward(Reward.INVALID_ACTION, "invalid trade")
            return

        match (trade_id):
            case TradeId.wool:
                if self.wool >= Restriction.MAX_WOOL:
                    invalid_action = True
                else:
                    self.wool += 1
                    self.emerald -= TRADE_COST[trade_id]
                    self.reward_collector.add_reward(Reward.TRADE_WOOL, "trade wool")
            case TradeId.life_potion:
                if self.hp >= self.hp_bound:
                    invalid_action = True
                else:
                    self.hp += 1
                    self.emerald -= TRADE_COST[trade_id]
                    self.reward_collector.add_reward(
                        Reward.TRADE_LIFE_POTION, "trade life potion"
                    )
            case TradeId.hp_limit_up:
                if self.hp_bound >= Restriction.MAX_HP:
                    invalid_action = True
                else:
                    self.hp_bound += 3
                    self.hp += 3
                    self.emerald -= TRADE_COST[trade_id]
                    self.reward_collector.add_reward(
                        Reward.TRADE_HP_LIMIT_UP, "trade hp limit up"
                    )
            case TradeId.atk_up:
                if self.atk >= Restriction.MAX_ATK:
                    invalid_action = True
                else:
                    self.atk += 1
                    self.emerald -= TRADE_COST[trade_id]
                    self.reward_collector.add_reward(
                        Reward.TRADE_ATK_UP, "trade atk up"
                    )
            case TradeId.haste_up:
                if self.haste >= Restriction.MAX_HASTE:
                    invalid_action = True
                else:
                    self.haste += 1
                    self.emerald -= TRADE_COST[trade_id]
                    self.reward_collector.add_reward(
                        Reward.TRADE_HASTE_UP, "trade haste up"
                    )

        if invalid_action:
            self.reward_collector.add_reward(Reward.INVALID_ACTION, "invalid trade")

    def collect_emerald(self, count):
        self.emerald += count

    def get_emerald_count(self):
        return self.emerald

    def play_action(self, action_id: ActionId, op: "Player"):
        match (action_id):
            case ActionId.DO_NOTHING:
                pass
            case ActionId.MOVE_F:
                self.move(Direction.FORWARD)
            case ActionId.MOVE_B:
                self.move(Direction.BACK)
            case ActionId.MOVE_L:
                self.move(Direction.LEFT)
            case ActionId.MOVE_R:
                self.move(Direction.RIGHT)
            case ActionId.DIVE_F:
                self.dive(Direction.FORWARD, op)
            case ActionId.DIVE_B:
                self.dive(Direction.BACK, op)
            case ActionId.DIVE_L:
                self.dive(Direction.LEFT, op)
            case ActionId.DIVE_R:
                self.dive(Direction.RIGHT, op)
            case ActionId.PLACE_F:
                self.place_block(Direction.FORWARD, op)
            case ActionId.PLACE_B:
                self.place_block(Direction.BACK, op)
            case ActionId.PLACE_L:
                self.place_block(Direction.LEFT, op)
            case ActionId.PLACE_R:
                self.place_block(Direction.RIGHT, op)
            case ActionId.ATTACK:
                self.attack(op)
            case ActionId.TRADE_WOOL:
                self.trade(TradeId.wool)
            case ActionId.HP_POTION:
                self.trade(TradeId.life_potion)
            case ActionId.HP_BOUND_UP:
                self.trade(TradeId.hp_limit_up)
            case ActionId.HASTE_UP:
                self.trade(TradeId.haste_up)
            case ActionId.ATK_UP:
                self.trade(TradeId.atk_up)
