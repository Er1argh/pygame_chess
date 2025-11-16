"""Утилиты для преобразования координат ввода (пиксели) в координаты доски.

Модуль содержит простую статическую функцию, которая переводит позицию
в пикселях экрана в индекс клетки доски (row, col) и при необходимости
ограничивает значения границами доски.
"""

from typing import Tuple

from src.config import SQUARE_SIZE, COLS, Pos, ROWS


class InputHandler:
    """Вспомогательные методы для обработки пользовательского ввода."""

    @staticmethod
    def pixel_to_square(pos: Tuple[int, int]) -> Pos:
        """Преобразовать пиксельные координаты в координаты клетки доски.

        Координата `pos` передаётся как (x, y) в пикселях — метод возвращает
        кортеж (row, col), где row — индекс строки (0..ROWS-1), col — индекс
        столбца (0..COLS-1). Значения вне окна автоматически обрезаются
        (clamped) к диапазону допустимых индексов.

        Args:
            pos: координаты пикселя в формате (x, y).

        Returns:
            Pos: кортеж (row, col) — индекс клетки на доске.

        Example:
            >>> InputHandler.pixel_to_square((10, 20))
            (0, 0)
        """
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        col = max(0, min(COLS - 1, col))
        row = max(0, min(ROWS - 1, row))

        return row, col
