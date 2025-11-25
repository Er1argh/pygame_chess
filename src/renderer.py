"""Визуализация доски, фигур и UI-элементов.

Модуль содержит класс `Renderer`, который умеет:
- загружать спрайты фигур из папки assets/sprites (с автоматическим масштабированием),
- отрисовывать доску, подсветку выбранной клетки и маркеры возможных ходов,
- рендерить движущиеся фигуры по их пиксельной позиции для анимации,
- показывать маленькое всплывающее окно выбора промоции пешки.

Рендерер предпочитает использовать спрайты; при их отсутствии отображаются
Unicode-символы, определённые в `UNICODE_PIECES`.
"""

import os
from typing import Dict, Tuple, Optional, List

import pygame

from src.anim import AnimatedMove
from src.config import Pos, DARK, SQUARE_SIZE, LIGHT, COLS, ROWS, HIGHLIGHT, MOVE_MARK, UNICODE_PIECES, BLACK_TEXT, \
    WIDTH, HEIGHT
from src.engine import Board, Piece


class Renderer:
    """Класс, отвечающий за всё, что связано с отрисовкой.

    Args:
        surface: Pygame surface, на который производится рисование.

    Атрибуты:
        surface: переданная Pygame поверхность.
        font: Pygame-шрифт для рендеринга текста/глифов.
        piece_images: словарь {(color, kind): pygame.Surface} с загруженными спрайтами.
    """

    def __init__(self, surface):
        self.surface = surface

        try:
            self.font = pygame.font.SysFont('segoeuisymbol', 36)
        except Exception:
            self.font = pygame.font.SysFont(None, 36)

        self.piece_images: Dict[Tuple[str, str], pygame.Surface] = {}
        self.load_piece_images()

    def load_piece_images(self, assets_dir: str = 'assets'):
        """Загрузить изображения фигур из `assets_dir/sprites/`.

        Файлы ожидаются по шаблону: `assets_dir/sprites/{color}{kind}.png`,
        где `color` — 'w' или 'b', `kind` — одна из 'K','Q','R','B','N','P'.

        Загруженные изображения масштабируются до (SQUARE_SIZE, SQUARE_SIZE)
        и сохраняются в `self.piece_images`. Ошибки при загрузке игнорируются —
        рендерер просто использует fallback (unicode) для таких фигур.
        """
        for color in ('w', 'b'):
            for kind in ('K', 'Q', 'R', 'B', 'N', 'P'):
                fname = os.path.join(assets_dir, f"sprites/{color}{kind}.png")

                if os.path.exists(fname):

                    try:
                        img = pygame.image.load(fname).convert_alpha()
                        img = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                        self.piece_images[(color, kind)] = img
                    except Exception:
                        pass

    def draw_board(self, board: Board, selected: Optional[Pos], legal_moves: List[Pos],
                   animated_move: Optional[AnimatedMove]):
        """Отрисовать клетки доски, подсветку, маркеры ходов и неперемещающиеся фигуры.

        Args:
            board: Текущее состояние доски (Board).
            selected: Выбранная клетка (row, col) или None — для подсветки.
            legal_moves: Список клеток-целей (row, col) для выбранной фигуры (для маркеров).
            animated_move: Объект AnimatedMove с информацией о движущихся фигурах.
                Фигуры, принимающие участие в анимации, не рисуются как статичные;
                их пиксельная позиция рассчитывается и отрисовывается далее.
        """
        for r in range(ROWS):
            for c in range(COLS):
                color = LIGHT if (r + c) % 2 == 0 else DARK
                rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.surface, color, rect)

        if selected:
            r, c = selected
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.surface, HIGHLIGHT, rect)

        # for (r, c) in legal_moves:
        #     center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
        #     pygame.draw.circle(self.surface, MOVE_MARK, center, 10)

        moving_from_set = set()

        if animated_move:
            for mp in animated_move.moving:
                moving_from_set.add(mp.from_sq)

        for r in range(ROWS):
            for c in range(COLS):
                pos = (r, c)

                if pos in moving_from_set:
                    continue

                p = board.get(pos)

                if p:
                    self.draw_piece_at_square(p, pos)

        for (r, c) in legal_moves:
            center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(self.surface, MOVE_MARK, center, 10)

        if animated_move:
            now_ms = pygame.time.get_ticks()
            t = animated_move.progress(now_ms)

            for mp in animated_move.moving:
                sx, sy = mp.from_sq[1] * SQUARE_SIZE + SQUARE_SIZE // 2, mp.from_sq[0] * SQUARE_SIZE + SQUARE_SIZE // 2
                ex, ey = mp.to_sq[1] * SQUARE_SIZE + SQUARE_SIZE // 2, mp.to_sq[0] * SQUARE_SIZE + SQUARE_SIZE // 2
                ix = sx + (ex - sx) * t
                iy = sy + (ey - sy) * t
                self.draw_piece_at_pixel(mp.piece, (int(ix), int(iy)))

    def draw_piece_at_square(self, piece: Piece, square: Pos):
        """Отрисовать `piece`, выровненный по клетке `square`.

        Если для фигуры есть спрайт в `self.piece_images`, он будет отрисован.
        В противном случае используется unicode-символ из UNICODE_PIECES
        или текстовое представление `piece.symbol()`.

        Args:
            piece: объект Piece.
            square: (row, col) — клетка, в которой рисуем фигуру.
        """
        r, c = square
        x = c * SQUARE_SIZE
        y = r * SQUARE_SIZE
        img = self.piece_images.get((piece.color, piece.kind))

        if img is not None:
            self.surface.blit(img, (x, y))
        else:
            ch = UNICODE_PIECES.get((piece.color, piece.kind), piece.symbol())
            text = self.font.render(ch, True, BLACK_TEXT)
            tx = x + SQUARE_SIZE // 2 - text.get_width() // 2
            ty = y + SQUARE_SIZE // 2 - text.get_height() // 2
            self.surface.blit(text, (tx, ty))

    def draw_piece_at_pixel(self, piece: Piece, pixel_center: Tuple[int, int]):
        """Отрисовать `piece`, центрированную в пиксельной позиции `pixel_center`.

        Используется для плавной анимации движения фигур.

        Args:
            piece: объект Piece.
            pixel_center: (x, y) — центр, в котором нужно нарисовать фигуру.
        """
        img = self.piece_images.get((piece.color, piece.kind))

        if img is not None:
            rect = img.get_rect()
            rect.center = pixel_center
            self.surface.blit(img, rect.topleft)
        else:
            ch = UNICODE_PIECES.get((piece.color, piece.kind), piece.symbol())
            text = self.font.render(ch, True, BLACK_TEXT)
            tx = pixel_center[0] - text.get_width() // 2
            ty = pixel_center[1] - text.get_height() // 2
            self.surface.blit(text, (tx, ty))

    def draw_promotion_popup(self, color: str, center: Tuple[int, int]) -> List[Tuple[pygame.Rect, str]]:
        """Показать popup выбора промоции пешки рядом с `center`.

        Рисует прямоугольники с возможными вариантами (Q, R, B, N). Для каждой
        опции возвращается соответствующий `pygame.Rect`, чтобы внешний код
        мог определить, какой вариант выбрал пользователь по клику.

        Args:
            color: 'w' или 'b' — цвет промоируемой пешки (для отображения правильных спрайтов/глифов).
            center: (x, y) — пиксельный центр, около которого нужно разместить popup.

        Returns:
            Список пар (pygame.Rect, choice_char), где choice_char ∈ {'Q','R','B','N'}.
        """
        choices = ['Q', 'R', 'B', 'N']
        size = 60
        spacing = 10
        margin = 8
        total_w = len(choices) * size + (len(choices) - 1) * spacing
        start_x = center[0] - total_w // 2
        y = center[1] - size // 2

        if start_x < margin:
            start_x = margin

        if start_x + total_w > WIDTH - margin:
            start_x = WIDTH - margin - total_w

        if y < margin:
            y = margin

        if y + size > HEIGHT - margin:
            y = HEIGHT - margin - size

        rects = []

        for i, ch in enumerate(choices):
            rect = pygame.Rect(int(start_x + i * (size + spacing)), int(y), size, size)
            pygame.draw.rect(self.surface, (220, 220, 220), rect)
            pygame.draw.rect(self.surface, (120, 120, 120), rect, 2)
            img = self.piece_images.get((color, ch))

            if img is not None:
                scaled = pygame.transform.smoothscale(img, (rect.width - 8, rect.height - 8))
                self.surface.blit(scaled, (rect.x + 4, rect.y + 4))
            else:
                symbol = UNICODE_PIECES.get((color, ch), ch)
                text = self.font.render(symbol, True, BLACK_TEXT)
                tx = rect.x + rect.width // 2 - text.get_width() // 2
                ty = rect.y + rect.height // 2 - text.get_height() // 2
                self.surface.blit(text, (tx, ty))

            rects.append((rect, ch))

        return rects

    def draw(self, board: Board, selected: Optional[Pos], legal_moves: List[Pos],
             animated_move: Optional[AnimatedMove] = None, promotion_center: Optional[Tuple[int, int]] = None,
             promotion_color: Optional[str] = None) -> Optional[List[Tuple[pygame.Rect, str]]]:
        """Высокоуровневая функция рисования кадра.

        Вызывает `draw_board` для основной отрисовки и при необходимости
        добавляет popup выбора промоции.

        Args:
            board: Текущее состояние доски.
            selected: Выбранная клетка (или None).
            legal_moves: Список возможных ходов для выбранной фигуры.
            animated_move: Объект анимации, если есть текущая анимация.
            promotion_center: Пиксельный центр для popup промоции (опционально).
            promotion_color: Цвет пешки, для которой открывается popup (опционально).

        Returns:
            Список (pygame.Rect, choice_char) при показанном popup, иначе None.
        """
        self.draw_board(board, selected, legal_moves, animated_move)

        if promotion_center and promotion_color:
            return self.draw_promotion_popup(promotion_color, promotion_center)

        return None
