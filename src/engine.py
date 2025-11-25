"""Движок игры: представление фигур, доски, ходов и генерация ходов.

Модуль содержит:
- базовый класс Piece и конкретные реализации фигур (Pawn, Knight, Bishop, Rook, Queen, King);
- dataclass Move для описания хода и вспомогательный PieceFactory;
- Board — контейнер 8x8 с методами мутации/отката хода и утилитами;
- GameEngine — верхнеуровневая логика: получение псевдо- и легальных ходов, проверка шаха.

Докстринги описывают поведение публичных методов и побочные эффекты
(например, что `make_move_and_return_captured` мутирует доску и заполняет поля
в объекте Move для возможности отката).
"""

from dataclasses import dataclass
from typing import List, Optional, Type, Dict

from src.config import Pos


class Piece:
    """Базовый класс для шахматной фигуры.

    Subclasses должны реализовать метод `pseudo_legal_moves`.

    Args:
        color: 'w' для белых или 'b' для чёрных.
        kind: одиночная буква, обозначающая тип фигуры ('P','N','B','R','Q','K').

    Attributes:
        has_moved: флаг, указывающий, делалась ли фигура.
    """

    def __init__(self, color: str, kind: str):
        assert color in ('w', 'b')
        self.color = color
        self.kind = kind
        self.has_moved = False

    def __repr__(self):
        return f"{self.color}{self.kind}"

    def symbol(self) -> str:
        """Возвращает текстовый символ для отрисовки в fallback-режиме."""
        return self.kind

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        """Вернуть список псевдо-легальных ходов для фигуры с позиции `pos`.

        Псевдо-легальные ходы соответствуют правилам перемещения фигуры и
        учитывают коллизии на доске, но не проверяют на шах после хода.

        Subclasses должны переопределять этот метод.
        """
        raise NotImplementedError


class Pawn(Piece):
    """Логика перемещения пешки: шаг вперёд, двойной шаг, захваты и en-passant."""

    def __init__(self, color: str):
        super().__init__(color, 'P')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        """Вычислить псевдо-легальные ходы пешки.

        Реализованы:
          - одиночный ход вперёд;
          - стартовый двойной ход (если обе клетки свободны);
          - диагональные захваты;
          - взятие на проходе при существующем board.en_passant_target.

        Args:
            board: объект Board.
            pos: текущее положение пешки (row, col).

        Returns:
            Список клеток (row, col), куда пешка может пойти (псевдо-легальные).
        """
        moves: List[Pos] = []
        r, c = pos
        step = -1 if self.color == 'w' else 1
        fr = r + step

        if board.in_bounds((fr, c)) and board.get((fr, c)) is None:
            moves.append((fr, c))
            start_row = 6 if self.color == 'w' else 1
            fr2 = r + 2 * step

            if r == start_row and board.in_bounds((fr2, c)) and board.get((fr2, c)) is None:
                moves.append((fr2, c))

        for dc in (-1, 1):
            tc = c + dc
            tr = r + step

            if board.in_bounds((tr, tc)):
                target = board.get((tr, tc))

                if target is not None and target.color != self.color:
                    moves.append((tr, tc))
                if board.en_passant_target is not None and board.en_passant_target == (tr, tc):
                    moves.append((tr, tc))

        return moves


class Knight(Piece):
    """Конь: прыжки L-образные, игнорирует промежуточные клетки."""

    def __init__(self, color: str):
        super().__init__(color, 'N')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        moves: List[Pos] = []
        r, c = pos
        deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

        for dr, dc in deltas:
            nr, nc = r + dr, c + dc

            if board.in_bounds((nr, nc)):
                target = board.get((nr, nc))

                if target is None or target.color != self.color:
                    moves.append((nr, nc))

        return moves


class Bishop(Piece):
    """Слон: скользящая диагональная фигура."""

    def __init__(self, color: str):
        super().__init__(color, 'B')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        moves: List[Pos] = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc

            while board.in_bounds((nr, nc)):
                target = board.get((nr, nc))

                if target is None:
                    moves.append((nr, nc))
                else:
                    if target.color != self.color:
                        moves.append((nr, nc))
                    break

                nr += dr
                nc += dc

        return moves


class Rook(Piece):
    """Ладья: скользящая ортогональная фигура."""

    def __init__(self, color: str):
        super().__init__(color, 'R')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        moves: List[Pos] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc

            while board.in_bounds((nr, nc)):
                target = board.get((nr, nc))

                if target is None:
                    moves.append((nr, nc))
                else:
                    if target.color != self.color:
                        moves.append((nr, nc))
                    break

                nr += dr
                nc += dc

        return moves


class Queen(Piece):
    """Ферзь: комбинация ходов ладьи и слона."""

    def __init__(self, color: str):
        super().__init__(color, 'Q')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        return Rook(self.color).pseudo_legal_moves(board, pos) + Bishop(self.color).pseudo_legal_moves(board, pos)


class King(Piece):
    """Король: одиночные ходы + первичная проверка рокировки (предлагает целевые клетки)."""

    def __init__(self, color: str):
        super().__init__(color, 'K')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        """Вернуть кандидатов ходов короля (не проверяет проход через шах — это делает GameEngine)."""
        moves: List[Pos] = []
        r, c = pos

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc

                if board.in_bounds((nr, nc)):
                    target = board.get((nr, nc))

                    if target is None or target.color != self.color:
                        moves.append((nr, nc))

        if not self.has_moved:
            row = r

            if board.get((row, 7)) is not None and isinstance(board.get((row, 7)), Rook) and not board.get(
                    (row, 7)).has_moved:
                if board.get((row, 5)) is None and board.get((row, 6)) is None:
                    moves.append((row, 6))
            if board.get((row, 0)) is not None and isinstance(board.get((row, 0)), Rook) and not board.get(
                    (row, 0)).has_moved:
                if board.get((row, 1)) is None and board.get((row, 2)) is None and board.get((row, 3)) is None:
                    moves.append((row, 2))

        return moves


@dataclass
class Move:
    """Лёгкая структура для описания хода.

    Поля используются как при применении хода (make_move_and_return_captured), так и
    при откате (unmake_move) — часть информации (prev_*) сохраняется на самом объекте Move.
    """
    from_sq: Pos
    to_sq: Pos
    piece: Piece
    captured: Optional[Piece] = None
    promotion: Optional[str] = None
    is_castling: bool = False
    castling_rook_from: Optional[Pos] = None
    castling_rook_to: Optional[Pos] = None
    is_en_passant: bool = False
    en_passant_captured_sq: Optional[Pos] = None
    prev_piece_has_moved: Optional[bool] = None
    prev_rook_has_moved: Optional[bool] = None
    prev_en_passant_target: Optional[Pos] = None


_piece_map: Dict[str, Type[Piece]] = {
    'P': Pawn, 'N': Knight, 'B': Bishop, 'R': Rook, 'Q': Queen, 'K': King
}


class PieceFactory:
    """Утилита для создания объектов Piece из символов FEN."""

    @staticmethod
    def piece_from_fen(ch: str) -> Piece:
        """Создать экземпляр фигуры для FEN-символа ``ch``.

        Примеры:
            'p' -> чёрная пешка, 'N' -> белый конь.

        Args:
            ch: символ FEN (буква в верхнем/нижнем регистре).

        Returns:
            Экземпляр соответствующего подкласса Piece.
        """
        color = 'w' if ch.isupper() else 'b'
        kind = ch.upper()
        cls = _piece_map[kind]
        return cls(color)


class Board:
    """8x8 контейнер доски с методами мутации/отката ходов и утилитами.

    Хранит grid (List[List[Optional[Piece]]]) индексируемую как [row][col],
    где 0..7 соответствуют строкам/столбцам.
    """
    ROWS = 8
    COLS = 8

    def __init__(self):
        self.grid: List[List[Optional[Piece]]] = [[None] * self.COLS for _ in range(self.ROWS)]
        self.en_passant_target: Optional[Pos] = None
        self.setup_start_position()

    def in_bounds(self, pos: Pos) -> bool:
        """Возвращает True, если `pos` лежит внутри доски."""
        r, c = pos
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def get(self, pos: Pos) -> Optional[Piece]:
        """Получить фигуру на `pos` или None."""
        r, c = pos
        return self.grid[r][c]

    def set(self, pos: Pos, piece: Optional[Piece]):
        """Поместить `piece` на `pos` (или очистить клетку, передав None)."""
        r, c = pos
        self.grid[r][c] = piece

    def setup_start_position(self):
        """Инициализация стартовой шахматной позиции (стандартная FEN-подстановка)."""
        start_fen = [
            "rnbqkbnr",
            "pppppppp",
            "........",
            "........",
            "........",
            "........",
            "PPPPPPPP",
            "RNBQKBNR",
        ]

        for r in range(8):
            row = start_fen[r]

            for c, ch in enumerate(row):
                if ch == '.':
                    self.grid[r][c] = None
                else:
                    self.grid[r][c] = PieceFactory.piece_from_fen(ch)

    def find_king(self, color: str) -> Optional[Pos]:
        """Найти позицию короля заданного `color` или вернуть None."""
        for r in range(self.ROWS):
            for c in range(self.COLS):
                p = self.grid[r][c]

                if p is not None and p.color == color and p.kind == 'K':
                    return r, c

        return None

    def make_move_and_return_captured(self, move: Move) -> Optional[Piece]:
        """Применить `move` к доске и вернуть захваченную фигуру (если была).

        Метод мутирует доску и записывает значения prev_* в сам объект `move`
        чтобы `unmake_move` мог восстановить состояние.

        Args:
            move: объект Move описывающий ход.

        Returns:
            captured: захваченная фигура или None.
        """
        move.prev_en_passant_target = self.en_passant_target
        move.prev_piece_has_moved = move.piece.has_moved
        self.en_passant_target = None

        if move.is_castling:
            self.set(move.to_sq, move.piece)
            self.set(move.from_sq, None)
            move.piece.has_moved = True
            rf = move.castling_rook_from
            rt = move.castling_rook_to
            rook = self.get(rf)
            move.prev_rook_has_moved = rook.has_moved if rook else None
            self.set(rt, rook)
            self.set(rf, None)

            if rook:
                rook.has_moved = True

            return None

        if move.is_en_passant and move.en_passant_captured_sq is not None:
            captured = self.get(move.en_passant_captured_sq)
            self.set(move.to_sq, move.piece)
            self.set(move.from_sq, None)
            self.set(move.en_passant_captured_sq, None)
            move.piece.has_moved = True

            return captured

        captured = self.get(move.to_sq)
        self.set(move.to_sq, move.piece)
        self.set(move.from_sq, None)

        if move.piece.kind == 'P':
            if abs(move.to_sq[0] - move.from_sq[0]) == 2:
                mid_row = (move.to_sq[0] + move.from_sq[0]) // 2
                self.en_passant_target = (mid_row, move.to_sq[1])
        if move.promotion and move.piece.kind == 'P':
            promo = move.promotion.upper()

            if promo == 'Q':
                self.set(move.to_sq, Queen(move.piece.color))
            elif promo == 'R':
                self.set(move.to_sq, Rook(move.piece.color))
            elif promo == 'B':
                self.set(move.to_sq, Bishop(move.piece.color))
            elif promo == 'N':
                self.set(move.to_sq, Knight(move.piece.color))
        else:
            move.piece.has_moved = True

        return captured

    def unmake_move(self, move: Move, captured: Optional[Piece]):
        """Откатить ранее применённый `move`, восстанавливая `captured` туда, где было.

        Args:
            move: объект Move, который был применён ранее.
            captured: значение, возвращённое make_move_and_return_captured для этого хода.
        """
        self.en_passant_target = move.prev_en_passant_target

        if move.is_castling:
            king = self.get(move.to_sq)
            self.set(move.from_sq, king)
            self.set(move.to_sq, None)

            if king:
                king.has_moved = move.prev_piece_has_moved

            rf = move.castling_rook_from
            rt = move.castling_rook_to
            rook = self.get(rt)
            self.set(rf, rook)
            self.set(rt, None)

            if rook:
                rook.has_moved = move.prev_rook_has_moved

            return

        if move.is_en_passant and move.en_passant_captured_sq is not None:
            pawn = self.get(move.to_sq)
            self.set(move.from_sq, pawn)
            self.set(move.to_sq, None)
            self.set(move.en_passant_captured_sq, captured)

            if pawn:
                pawn.has_moved = move.prev_piece_has_moved

            return

        if move.promotion and move.piece.kind == 'P':
            self.set(move.to_sq, captured)
            self.set(move.from_sq, move.piece)
            move.piece.has_moved = move.prev_piece_has_moved

            return

        self.set(move.from_sq, move.piece)
        self.set(move.to_sq, captured)
        move.piece.has_moved = move.prev_piece_has_moved

    def is_square_attacked(self, square: Pos, by_color: str) -> bool:
        """Проверить, атакуется ли `square` стороной `by_color`.

        Учитываются атаки пешек, коней, скользящих фигур (ладья/слон/ферзь) и короля.
        Метод не изменяет состояние доски.
        """
        r, c = square
        pawn_dir = -1 if by_color == 'w' else 1

        for dc in (-1, 1):
            pr, pc = r - pawn_dir, c - dc

            if self.in_bounds((pr, pc)):
                p = self.get((pr, pc))

                if p is not None and p.color == by_color and p.kind == 'P':
                    return True

        knight_deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

        for dr, dc in knight_deltas:
            nr, nc = r + dr, c + dc

            if self.in_bounds((nr, nc)):
                p = self.get((nr, nc))

                if p is not None and p.color == by_color and p.kind == 'N':
                    return True

        straight_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in straight_dirs:
            nr, nc = r + dr, c + dc

            while self.in_bounds((nr, nc)):
                p = self.get((nr, nc))

                if p is not None:
                    if p.color == by_color and (p.kind == 'R' or p.kind == 'Q'):
                        return True

                    break

                nr += dr
                nc += dc

        diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in diag_dirs:
            nr, nc = r + dr, c + dc

            while self.in_bounds((nr, nc)):
                p = self.get((nr, nc))

                if p is not None:
                    if p.color == by_color and (p.kind == 'B' or p.kind == 'Q'):
                        return True

                    break

                nr += dr
                nc += dc

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue

                nr, nc = r + dr, c + dc

                if self.in_bounds((nr, nc)):
                    p = self.get((nr, nc))

                    if p is not None and p.color == by_color and p.kind == 'K':
                        return True

        return False

    def fen(self) -> str:
        """Вернуть строку, похожую на часть FEN — только размещение фигур по строкам."""
        parts = []

        for r in range(8):
            empty = 0
            rowp = ""

            for c in range(8):
                p = self.get((r, c))

                if p is None:
                    empty += 1
                else:
                    if empty:
                        rowp += str(empty)
                        empty = 0

                    ch = p.kind.upper() if p.color == 'w' else p.kind.lower()
                    rowp += ch

            if empty:
                rowp += str(empty)

            parts.append(rowp)

        return "/".join(parts)

    def __str__(self):
        """Человеко-читаемое ASCII-представление для REPL / тестов."""
        rows = []

        for r in range(8):
            row = []

            for c in range(8):
                p = self.get((r, c))
                row.append(str(p) if p else "..")

            rows.append(" ".join(row))

        return "\n".join(rows)


class GameEngine:
    """Верхнеуровневая логика игры: филтрация псевдо-ходов, проверка шаха, сбор легальных ходов."""

    def __init__(self, board: Optional[Board] = None):
        """Инициализировать движок с указанной доской или новой по умолчанию."""
        self.board = board if board is not None else Board()
        self.turn = 'w'

    def all_pseudo_legal_moves_for(self, color: str) -> List[Move]:
        """Собрать все псевдо-легальные ходы для стороны `color`.

        Возвращает список объектов Move. Устанавливает флаги is_castling и is_en_passant
        при обнаружении таких ходов.
        """
        moves: List[Move] = []

        for r in range(8):
            for c in range(8):
                p = self.board.get((r, c))

                if p is not None and p.color == color:
                    for to_sq in p.pseudo_legal_moves(self.board, (r, c)):
                        mv = Move((r, c), to_sq, p, self.board.get(to_sq))

                        if p.kind == 'K' and abs(to_sq[1] - c) == 2:
                            mv.is_castling = True
                            row = r

                            if to_sq[1] > c:
                                mv.castling_rook_from = (row, 7)
                                mv.castling_rook_to = (row, 5)
                            else:
                                mv.castling_rook_from = (row, 0)
                                mv.castling_rook_to = (row, 3)

                        if p.kind == 'P' and self.board.en_passant_target is not None and to_sq == self.board.en_passant_target:
                            mv.is_en_passant = True
                            mv.en_passant_captured_sq = (r, to_sq[1])

                        moves.append(mv)

        return moves

    def is_in_check(self, color: str) -> bool:
        """Вернуть True, если сторона `color` находится под шахом.

        Если король отсутствует на доске — возвращает False (удобно для тестов).
        """
        king_pos = self.board.find_king(color)

        if king_pos is None:
            return False

        opponent = 'b' if color == 'w' else 'w'

        return self.board.is_square_attacked(king_pos, opponent)

    def is_move_legal(self, move: Move) -> bool:
        """Проверить, является ли `move` легальным (не оставляет короля в шахе).

        Алгоритм:
          - валидация согласованности объекта Move с текущей доской;
          - проверка, что цель входит в pseudo_legal_moves;
          - для рокировки дополнительно проверяется, что король не под шахом и не проходит через атакуемые клетки;
          - временный ход применяется (make_move_and_return_captured), затем проверяется шах, и состояние откатывается.

        Возвращает True, если ход легален.
        """
        p = self.board.get(move.from_sq)

        if p is None or p.color != move.piece.color or p.kind != move.piece.kind:
            return False

        if move.to_sq not in move.piece.pseudo_legal_moves(self.board, move.from_sq):
            return False

        if move.is_castling:
            if self.is_in_check(move.piece.color):
                return False

            r, c_from = move.from_sq
            c_to = move.to_sq[1]
            step = 1 if c_to > c_from else -1
            opponent = 'b' if move.piece.color == 'w' else 'w'

            for c in (c_from + step, c_from + 2 * step):
                if self.board.is_square_attacked((r, c), opponent):
                    return False

        captured = self.board.make_move_and_return_captured(move)
        in_check = self.is_in_check(move.piece.color)
        self.board.unmake_move(move, captured)

        return not in_check

    def legal_moves_for(self, pos: Pos) -> List[Pos]:
        """Вернуть список легальных целевых клеток для фигуры на `pos`.

        Фильтрует псевдо-ходы через `is_move_legal`.
        """

        p = self.board.get(pos)

        if p is None:
            return []

        pseudo = p.pseudo_legal_moves(self.board, pos)
        legal: List[Pos] = []

        for to in pseudo:
            mv = Move(pos, to, p, self.board.get(to))

            if p.kind == 'K' and abs(to[1] - pos[1]) == 2:
                mv.is_castling = True
                row = pos[0]

                if to[1] > pos[1]:
                    mv.castling_rook_from = (row, 7)
                    mv.castling_rook_to = (row, 5)
                else:
                    mv.castling_rook_from = (row, 0)
                    mv.castling_rook_to = (row, 3)

            if p.kind == 'P' and self.board.en_passant_target is not None and to == self.board.en_passant_target:
                mv.is_en_passant = True
                mv.en_passant_captured_sq = (pos[0], to[1])

            if self.is_move_legal(mv):
                legal.append(to)

        return legal
