import pygame
from typing import Union
import numpy as np

from globalConst import Restriction
from structs import Pos, Mine, Direction, PlayerId
from Vein import Vein
from Player import Player
from Settings import FPS, DEAD_MATCH_COUNTDOWN

BOX_SIZE = 70
COLOR_A = (0, 255, 255)
COLOR_B = (255, 255, 0)


class MapBoard:
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.canva = pygame.Surface((8 * BOX_SIZE, 8 * BOX_SIZE))
        self.block_gradient_colors: list[tuple[int, int, int]] = (
            self._generate_gradient_colors(
                (1, 50, 32), (188, 184, 138), Restriction.MAX_BLOCK_HEIGHT + 1
            )[::-1]
        )

    def _get_box_left_top(self, r, c):
        return (
            c * BOX_SIZE,
            r * BOX_SIZE,
        )

    def _draw_grid(self, color: Union[tuple[int, int, int], str]):
        GRID_WEIGHT = 1
        for r in range(8):
            for c in range(8):
                x, y = self._get_box_left_top(r, c)
                pygame.draw.rect(
                    self.canva, color, (x, y, BOX_SIZE, BOX_SIZE), GRID_WEIGHT
                )

    def _generate_gradient_colors(self, start_color, end_color, steps):
        """return n = steps colors between start_color and end_color"""
        gradient = []
        for i in range(steps):
            intermediate_color = (
                start_color[0] + (end_color[0] - start_color[0]) * i // (steps - 1),
                start_color[1] + (end_color[1] - start_color[1]) * i // (steps - 1),
                start_color[2] + (end_color[2] - start_color[2]) * i // (steps - 1),
            )
            gradient.append(intermediate_color)
        return gradient

    def _draw_map_blocks_color(self, height_map: np.ndarray):
        for r in range(8):
            for c in range(8):
                x, y = self._get_box_left_top(r, c)
                block_height = height_map[r, c]
                if block_height > len(self.block_gradient_colors) - 1:
                    color = self.block_gradient_colors[-1]
                else:
                    color = self.block_gradient_colors[block_height]
                pygame.draw.rect(self.canva, color, (x, y, BOX_SIZE, BOX_SIZE))

    def _draw_height_text(self, height_map, color: Union[tuple[int, int, int], str]):
        for r in range(8):
            for c in range(8):
                x, y = self._get_box_left_top(r, c)
                text = self.font.render(str(height_map[r, c]), True, color)
                self.canva.blit(text, (x + 4, y + 4))

    def _draw_vein_text(self, r, c, count, color):
        x, y = self._get_box_left_top(r, c)
        text = self.font.render(str(count), True, color)
        x = x + BOX_SIZE * 0.8
        y = y + BOX_SIZE * 0.7
        if count >= 10:
            x -= 10
        self.canva.blit(text, (x, y))

    def _draw_vein_color(self, r, c, mine: Mine):
        x, y = self._get_box_left_top(r, c)
        color = None
        if mine == Mine.iron:
            color = "darkgray"
        elif mine == Mine.gold:
            color = "gold1"
        elif mine == Mine.diamond:
            color = (135, 206, 250)
        assert color is not None
        pygame.draw.rect(self.canva, color, (x, y, BOX_SIZE, BOX_SIZE))

    def _draw_player(
        self,
        r: int,
        c: int,
        color: Union[tuple[int, int, int], str],
        heading: Direction,
        right: bool,
        draw_direction: bool,
    ):
        """
        Args:
            r: row
            c: col
            color: player color
            heading: player heading
            right: draw player in block right side or left side
            draw_direction: draw direction or not
        """
        radius = BOX_SIZE / 4
        x, y = self._get_box_left_top(r, c)
        x = round(x + 0.5 * (right + 0.5) * BOX_SIZE)
        y = round(y + BOX_SIZE / 2)
        pygame.draw.circle(self.canva, color, (x, y), radius)

        if not draw_direction:
            return

        DIRECTION_REMARK_COLOR = (255, 0, 0)
        label_color = DIRECTION_REMARK_COLOR

        if heading == Direction.CENTER:
            pygame.draw.circle(self.canva, label_color, (x, y), 0.4 * radius)
            return

        triangle_pos = []
        radius = round(radius * 0.9)
        match (heading):
            case Direction.FORWARD:
                triangle_pos = [
                    (x, y - radius),
                    (x + radius * pow(3, 0.5) / 2, y + radius / 2),
                    (x - radius * pow(3, 0.5) / 2, y + radius / 2),
                ]
            case Direction.BACK:
                triangle_pos = [
                    (x, y + radius),
                    (x + radius * pow(3, 0.5) / 2, y - radius / 2),
                    (x - radius * pow(3, 0.5) / 2, y - radius / 2),
                ]
            case Direction.LEFT:
                triangle_pos = [
                    (x - radius, y),
                    (x + radius / 2, y + radius * pow(3, 0.5) / 2),
                    (x + radius / 2, y - radius * pow(3, 0.5) / 2),
                ]
            case Direction.RIGHT:
                triangle_pos = [
                    (x + radius, y),
                    (x - radius / 2, y + radius * pow(3, 0.5) / 2),
                    (x - radius / 2, y - radius * pow(3, 0.5) / 2),
                ]
        assert triangle_pos
        pygame.draw.polygon(self.canva, label_color, triangle_pos)

    def update(
        self,
        height_map: np.ndarray,
        veins: list[Vein],
        player_A_pos: Pos,
        heading_A: Direction,
        player_B_pos: Pos,
        heading_B: Direction,
    ):
        TEXT_COLOR = (245, 245, 245)
        BG_COLOR = (10, 10, 10)
        GRID_COLOR = (10, 10, 10)

        self.canva.fill(BG_COLOR)

        self._draw_map_blocks_color(height_map)

        # draw veins
        for vein in veins:
            self._draw_vein_color(vein.pos.r, vein.pos.c, vein.type)
            self._draw_vein_text(vein.pos.r, vein.pos.c, vein.mine_counts, TEXT_COLOR)

        self._draw_height_text(height_map, TEXT_COLOR)

        # draw players
        if not player_A_pos.is_invalid():
            self._draw_player(
                r=player_A_pos.r,
                c=player_A_pos.c,
                color=COLOR_A,
                heading=heading_A,
                right=False,
                draw_direction=True,
            )
        if not player_B_pos.is_invalid():
            self._draw_player(
                r=player_B_pos.r,
                c=player_B_pos.c,
                color=COLOR_B,
                heading=heading_B,
                right=True,
                draw_direction=True,
            )

        self._draw_grid(GRID_COLOR)

        return self.get_canva()

    def get_canva(self):
        return self.canva


class DetailBoard:
    WIDTH = 480
    HEIGHT = 500

    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.canva = pygame.Surface((self.WIDTH, self.HEIGHT))

    def _draw_color_box(self, color, x, y):
        pygame.draw.rect(self.canva, color, (x, y, 30, 30))

    def _draw_text(self, text, x, y):
        img = self.font.render(text, True, (245, 245, 245))
        self.canva.blit(img, (x, y))

    def _draw_player_detail(self, x, y, player: Player):
        if player.id == PlayerId.Player_A:
            color = COLOR_A
        else:
            color = COLOR_B

        self._draw_color_box(color, x, y)
        self._draw_text(f"{player.id} : ", x + 40, y)
        text_list = [
            f"emerald: {player.emerald}",
            f"hp: {player.hp}  atk: {player.atk}",
            f"hp_bound: {player.hp_bound}",
            f"haste: {player.haste}",
            f"wool: {player.wool}  Cd:{round(player.calc_cd(player.haste),2)}",
            f"bed destroyed: {player.bed.is_destroyed()}",
            f"isAlive: {player.is_alive()}",
            f"inCoolDown: {player.is_in_cd()}",
        ]
        line = len(text_list)
        y_gap = 40
        _y = y
        for i in range(line):
            _y += y_gap
            self._draw_text(text_list[i], x, _y)

    def update(self, player_A: Player, player_B: Player, tick: int):
        self.canva.fill((10, 10, 10))
        pygame.draw.rect(
            self.canva, (255, 255, 255), (0, 0, self.WIDTH, self.HEIGHT), 2
        )
        self._draw_player_detail(10, 10, player_A)
        self._draw_player_detail(260, 10, player_B)
        self._draw_text(f"tick: {tick}", 10, 380)
        self._draw_text(f"game time: {tick // FPS}", 10, 420)
        self._draw_text(
            f"dead match started: {tick >= DEAD_MATCH_COUNTDOWN*FPS}", 10, 460
        )
        return self.get_canva()

    def get_canva(self):
        return self.canva
