import pygame
import numpy as np

from game.TickTimer import TickTimer
from game.Vein import Vein
from game.structs import Pos, Mine, TradeId
from game.structs import Direction, ActionId, PlayerObservation
from game.globalConst import Restriction, TRADE_COST
import game.Settings as Settings


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


def get_valid_actions_mask(obs: PlayerObservation):
    valid_actions_mask: np.ndarray[np.bool_] = np.full(19, False).astype(np.bool_)

    valid_actions_mask[ActionId.DO_NOTHING] = True

    if obs.hp <= 0 or obs.pos.is_invalid():
        return valid_actions_mask

    self_pos = obs.pos
    op_pos = obs.op_pos
    height_map = obs.height_map

    if self_pos.r > 0:
        # check if forward action is valid
        f_block_height = height_map[self_pos.r - 1, self_pos.c]
        if Settings.MOVE_ZERO_AVAILABLE or f_block_height > 0:
            valid_actions_mask[ActionId.MOVE_F] = True
        if f_block_height > 0 and not obs.in_cd:
            valid_actions_mask[ActionId.DIVE_F] = True
        if f_block_height < Restriction.MAX_BLOCK_HEIGHT and obs.wool > 0:
            valid_actions_mask[ActionId.PLACE_F] = True

    if self_pos.r < 7:
        # check if back action is valid
        b_block_height = height_map[self_pos.r + 1, self_pos.c]
        if Settings.MOVE_ZERO_AVAILABLE or b_block_height > 0:
            valid_actions_mask[ActionId.MOVE_B] = True
        if b_block_height > 0 and not obs.in_cd:
            valid_actions_mask[ActionId.DIVE_B] = True
        if b_block_height < Restriction.MAX_BLOCK_HEIGHT and obs.wool > 0:
            valid_actions_mask[ActionId.PLACE_B] = True

    if self_pos.c > 0:
        # check if left action is valid
        l_block_height = height_map[self_pos.r, self_pos.c - 1]
        if Settings.MOVE_ZERO_AVAILABLE or l_block_height > 0:
            valid_actions_mask[ActionId.MOVE_L] = True
        if l_block_height > 0 and not obs.in_cd:
            valid_actions_mask[ActionId.DIVE_L] = True
        if l_block_height < Restriction.MAX_BLOCK_HEIGHT and obs.wool > 0:
            valid_actions_mask[ActionId.PLACE_L] = True

    if self_pos.c < 7:
        # check if right action is valid
        r_block_height = height_map[self_pos.r, self_pos.c + 1]
        if Settings.MOVE_ZERO_AVAILABLE or r_block_height > 0:
            valid_actions_mask[ActionId.MOVE_R] = True
        if r_block_height > 0 and not obs.in_cd:
            valid_actions_mask[ActionId.DIVE_R] = True
        if r_block_height < Restriction.MAX_BLOCK_HEIGHT and obs.wool > 0:
            valid_actions_mask[ActionId.PLACE_R] = True

    # op is alive and adjacent, self not in cd
    if not op_pos.is_invalid() and self_pos.is_adjacent(op_pos) and not obs.in_cd:
        valid_actions_mask[ActionId.ATTACK] = True

    # check if trade is valid
    if self_pos == obs.bed_pos and (
        obs.half_second < Settings.DEAD_MATCH_COUNTDOWN * 2
        or Settings.DEAD_MATCH_TRADE_VALID
    ):
        if obs.emerald >= TRADE_COST[TradeId.wool] and obs.wool < Restriction.MAX_WOOL:
            valid_actions_mask[ActionId.TRADE_WOOL] = True
        if obs.emerald >= TRADE_COST[TradeId.life_potion] and obs.hp < obs.hp_bound:
            valid_actions_mask[ActionId.HP_POTION] = True
        if (
            obs.emerald >= TRADE_COST[TradeId.hp_limit_up]
            and obs.hp_bound < Restriction.MAX_HP
        ):
            valid_actions_mask[ActionId.HP_BOUND_UP] = True
        if obs.emerald >= TRADE_COST[TradeId.atk_up] and obs.atk < Restriction.MAX_ATK:
            valid_actions_mask[ActionId.ATK_UP] = True
        if (
            obs.emerald >= TRADE_COST[TradeId.haste_up]
            and obs.haste < Restriction.MAX_HASTE
        ):
            valid_actions_mask[ActionId.HASTE_UP] = True

    return valid_actions_mask
