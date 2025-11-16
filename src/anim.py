"""Примитивы анимации, используемые рендерером для плавного перемещения фигур.

Модуль содержит лёгкие структуры данных `MovingPiece` и `AnimatedMove`,
которые помогают хранить информацию о фигурах, участвующих в анимации, и
контролировать прогресс анимации во времени.

Ожидается, что время передаётся в миллисекундах и получаются через
`pygame.time.get_ticks()` в коде, который использует `AnimatedMove.progress`.
"""

from dataclasses import dataclass
from typing import List

from src.config import Pos, ANIMATION_DURATION
from src.engine import Piece, Move


@dataclass
class MovingPiece:
    """Временная структура, описывающая одну движущуюся фигуру.

    Attributes:
        piece: Ссылка на объект Piece, который движется.
        from_sq: Исходная клетка (row, col).
        to_sq: Конечная клетка (row, col).
    """
    piece: Piece
    from_sq: Pos
    to_sq: Pos


@dataclass
class AnimatedMove:
    """Представление текущей (выполняющейся) анимации хода.

    AnimatedMove хранит список объектов MovingPiece (один или несколько —
    например, король и ладья при рокировке), оригинальный объект Move,
    время старта и длительность анимации.

    Attributes:
        moving: Список MovingPiece, участвующих в анимации.
        move_obj: Исходный объект Move — применяется к доске по завершении.
        start_time_ms: Время старта анимации в миллисекундах (как у pygame.get_ticks()).
        duration_ms: Длительность анимации в миллисекундах (по умолчанию ANIMATION_DURATION).
        finished: Флаг, помечающий, что анимация завершилась (progress >= 1.0).
    """
    moving: List[MovingPiece]
    move_obj: Move
    start_time_ms: int
    duration_ms: int = ANIMATION_DURATION
    finished: bool = False

    def progress(self, now_ms: int) -> float:
        """Вычислить прогресс анимации в диапазоне [0.0, 1.0].

        Args:
            now_ms: Текущее время в миллисекундах (обычно pygame.time.get_ticks()).

        Returns:
            float: значение прогресса анимации, где 0.0 — только старт, 1.0 — завершено.

        Side effects:
            Устанавливает `self.finished = True`, если прошло времени >= duration_ms.
        """
        elapsed = now_ms - self.start_time_ms
        t = min(1.0, max(0.0, elapsed / self.duration_ms))

        if elapsed >= self.duration_ms:
            self.finished = True

        return t
