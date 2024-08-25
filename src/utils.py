from gymnasium.envs.registration import register
from game import BedWarGame

def register_game():
    register(
        id="BedWarGame-v0",
        entry_point="game:BedWarGame",
        additional_wrappers=[],
    )


