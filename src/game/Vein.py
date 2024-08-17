from game.structs import Pos, Mine
from game.TickTimer import TickTimer
import game.Settings as Settings


class Vein:
    def __init__(self, mine: Mine, pos: Pos, tick_timer: TickTimer):
        self.pos = pos
        self.type = mine
        self.mine_counts = 0
        self.timer = tick_timer.add_timer(
            Settings.VEIN_PRODUCE_CYCLE,
            self.produce_vein,
            forever=True,
            info=f"{mine} vein timer",
        )

    def produce_vein(self):
        self.mine_counts += 1

    def stop_produce(self):
        self.timer.terminate()


class Bed(Vein):
    def __init__(self, pos: Pos, tick_timer: TickTimer) -> None:
        super().__init__(Mine.iron, pos, tick_timer)
        self.destroyed = False
        self.trade_available = True

    def set_is_trade_available(self, is_trade_available: bool):
        self.trade_available = is_trade_available

    def set_destroyed(self):
        self.destroyed = True

    def is_destroyed(self):
        return self.destroyed

    def is_trade_available(self):
        return self.trade_available
