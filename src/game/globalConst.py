from game.structs import TradeId


class Restriction:
    MAX_WOOL = 64
    MAX_EMERALD = 128
    MAX_HP = 110
    MAX_ATK = 25
    MAX_HASTE = 32
    MAX_BLOCK_HEIGHT = 15
    MAX_TRAINING_TIME = 15 * 60  # 15 minutes
    """max training time in seconds to give truncated signal to the agent"""
    IS_DONE_IF_TIME_EXCEED = True
    """if True, the game is done if training time exceeds MAX_TRAINING_TIME"""


TRADE_COST = {
    TradeId.wool: 2,
    TradeId.life_potion: 4,
    TradeId.hp_limit_up: 32,
    TradeId.atk_up: 64,
    TradeId.haste_up: 32,
}


class Reward:
    TIE = 0
    WIN = 0
    LOSE = 0

    INVALID_ACTION = -0.2
    STEP_PENALTY = -0.002
    PER_EMERALD_GAIN = 0.8
    CLOSE_TO_VEINS = 0.01
    HOLD_WOOL = 0.002

    PER_DAMAGE_DEAL = 1
    KILL_OP = 20
    CONSTRUCT_BED = 0  # 0.15
    ATTACK_SELF_BED = -4
    ATTACK_BED = 1
    DESTROY_BED = 30

    PER_DAMAGE_TAKE = 0  # -0.4
    DEATH = -10
    BED_DESTROYED = -20
    BED_ATTACKED = 0  # -0.4

    TRADE_WOOL = 0.5
    TRADE_LIFE_POTION = 0.5
    TRADE_HP_LIMIT_UP = 8
    TRADE_ATK_UP = 16
    TRADE_HASTE_UP = 8
