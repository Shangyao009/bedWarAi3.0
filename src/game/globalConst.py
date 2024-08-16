from structs import TradeId

class Restriction:
    MAX_WOOL = 64
    MAX_EMERALD = 128
    MAX_HP = 110
    MAX_ATK = 25
    MAX_HASTE = 32
    MAX_BLOCK_HEIGHT = 15


TRADE_COST = {
    TradeId.wool: 2,
    TradeId.life_potion: 4,
    TradeId.hp_limit_up: 32,
    TradeId.atk_up: 64,
    TradeId.haste_up: 32,
}

class Reward:
    pass