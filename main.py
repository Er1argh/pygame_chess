"""
Chess (Pygame) — integrated chess engine with legal move generation and basic UI.


Features:
- Piece movement rules for P, N, B, R, Q, K
- Pseudo-legal -> legal move filtering (king safety / check detection)
- Click to select and move pieces (only legal moves allowed)
- Highlight selected square and legal move targets
- Undo (press 'u')
- Automatic pawn promotion to Queen when reaching last rank


Not implemented yet:
- Castling
- En-passant
- Promotion choice (auto-queen)


Run: pip install pygame ; python main.py
Controls:
- Left click: select piece / move to highlighted square
- U key: undo last move


This is a single-file minimal project, suitable for iterating and producing UML diagrams.
"""

import pygame
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Type, Dict

# Config
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
FPS = 60

LIGHT = (245, 245, 220)
DARK = (118, 150, 86)
HIGHLIGHT = (186, 202, 68)
MOVE_MARK = (200, 50, 50)
BLACK_TEXT = (20, 20, 20)

UNICODE_PIECES = {
    ('w', 'K'): '♔', ('w', 'Q'): '♕', ('w', 'R'): '♖', ('w', 'B'): '♗', ('w', 'N'): '♘', ('w', 'P'): '♙',
    ('b', 'K'): '♚', ('b', 'Q'): '♛', ('b', 'R'): '♜', ('b', 'B'): '♝', ('b', 'N'): '♞', ('b', 'P'): '♟',
}

Pos = Tuple[int, int]


# Engine: pieces, board, engine
class Piece:
    def __init__(self, color: str, kind: str):
        assert color in ('w', 'b')
        self.color = color  # 'w' | 'b'
        self.kind = kind  # 'K' | 'Q' | 'R' | 'B' | 'N' | 'P'

    def __repr__(self):
        # Цвет и тип фигуры (wK, например)
        return f"{self.color}{self.kind}"

    def symbol(self) -> str:
        # Текстовая метка для отрисовки (только тип фигуры без цвета)
        return self.kind

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        raise NotImplementedError


class Pawn(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'P')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        moves: List[Pos] = []
        r, c = pos
        dir = -1 if self.color == 'w' else 1
        fr = r + dir
        # forward
        if board.in_bounds((fr, c)) and board.get((fr, c)) is None:
            moves.append((fr, c))
            start_row = 6 if self.color == 'w' else 1
            fr2 = r + 2 * dir
            if r == start_row and board.in_bounds((fr2, c)) and board.get((fr2, c)) is None:
                moves.append((fr2, c))
        # captures
        for dc in (-1, 1):
            tc = c + dc
            tr = r + dir
            if board.in_bounds((tr, tc)):
                target = board.get((tr, tc))
                if target is not None and target.color != self.color:
                    moves.append((tr, tc))
        return moves


class Rook(Piece):
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


class Knight(Piece):
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


class Queen(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'Q')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
        return Rook(self.color).pseudo_legal_moves(board, pos) + Bishop(self.color).pseudo_legal_moves(board, pos)


class King(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'K')

    def pseudo_legal_moves(self, board: "Board", pos: Pos) -> List[Pos]:
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
        return moves


_piece_map: Dict[str, Type[Piece]] = {
    'P': Pawn, 'N': Knight, 'B': Bishop, 'R': Rook, 'Q': Queen, 'K': King
}


def piece_from_fen(ch: str) -> Piece:
    color = 'w' if ch.isupper() else 'b'
    kind = ch.upper()
    cls = _piece_map[kind]
    return cls(color)


# class PieceFactory:
#     # Создает фигуру из буквенных обозначений (большая буква - белый цвет, маленькая - черный)
#     @staticmethod
#     def from_fen_char(ch: str) -> Piece:
#         color = 'w' if ch.isupper() else 'b'
#         kind = ch.upper()
#         mapping = {'P': Pawn, 'R': Rook, 'N': Knight, 'B': Bishop, 'Q': Queen, 'K': King}
#         cls = mapping[kind]
#         return cls(color)


@dataclass
class Move:
    from_sq: Pos
    to_sq: Pos
    piece: Piece
    captured: Optional[Piece] = None
    promotion: Optional[str] = None


class Board:
    ROWS = 8
    COLS = 8

    def __init__(self):
        # grid[row][col]; row 0 - top (8th rank), row 7 - bottom (1st rank)
        self.grid: List[List[Optional[Piece]]] = [[None] * self.COLS for _ in range(self.ROWS)]
        self.setup_start_position()

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def get(self, pos: Pos) -> Optional[Piece]:
        r, c = pos
        return self.grid[r][c]

    def set(self, pos: Pos, piece: Optional[Piece]):
        r, c = pos
        self.grid[r][c] = piece

    def setup_start_position(self):
        # Простая расстановка: белые внизу (rows 6-7), чёрные вверху (rows 0-1)
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
                    self.grid[r][c] = piece_from_fen(ch)

    def find_king(self, color: str) -> Optional[Pos]:
        for r in range(self.ROWS):
            for c in range(self.COLS):
                p = self.grid[r][c]
                if p is not None and p.color == color and p.kind == 'K':
                    return (r, c)
        return None

    def make_move_and_return_captured(self, move: Move) -> Optional[Piece]:
        captured = self.get(move.to_sq)
        # move piece object reference
        self.set(move.to_sq, move.piece)
        self.set(move.from_sq, None)
        # handle promotion if requested
        if move.promotion and move.piece.kind == 'P':
            # simple promotion: replace by piece type (Q/R/B/N) — we only support Q for now
            promo = move.promotion.upper()
            if promo == 'Q':
                self.set(move.to_sq, Queen(move.piece.color))
        return captured

    def unmake_move(self, move: Move, captured: Optional[Piece]):
        # revert promotion by restoring original pawn if needed
        # if there was promotion we will assume original was a pawn
        if move.promotion and move.piece.kind == 'P':
            # restore pawn at from_sq
            self.set(move.from_sq, move.piece)
            self.set(move.to_sq, captured)
        else:
            self.set(move.from_sq, move.piece)
            self.set(move.to_sq, captured)

    def is_square_attacked(self, square: Pos, by_color: str) -> bool:
        r, c = square
        # pawn attacks
        pawn_dir = -1 if by_color == 'w' else 1
        for dc in (-1, 1):
            pr, pc = r - pawn_dir, c - dc
            if self.in_bounds((pr, pc)):
                p = self.get((pr, pc))
                if p is not None and p.color == by_color and p.kind == 'P':
                    return True
        # knight
        knight_deltas = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for dr, dc in knight_deltas:
            nr, nc = r + dr, c + dc
            if self.in_bounds((nr, nc)):
                p = self.get((nr, nc))
                if p is not None and p.color == by_color and p.kind == 'N':
                    return True
        # rook/queen straight
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
        # bishop/queen diagonals
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
        # king adjacency
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
        rows = []
        for r in range(8):
            row = []
            for c in range(8):
                p = self.get((r, c))
                row.append(str(p) if p else "..")
            rows.append(" ".join(row))
        return "\n".join(rows)


class GameEngine:
    def __init__(self, board: Optional[Board] = None):
        self.board = board if board is not None else Board()
        self.turn = 'w'

    def all_pseudo_legal_moves_for(self, color: str) -> List[Move]:
        moves: List[Move] = []
        for r in range(8):
            for c in range(8):
                p = self.board.get((r, c))
                if p is not None and p.color == color:
                    for to_sq in p.pseudo_legal_moves(self.board, (r, c)):
                        moves.append(Move((r, c), to_sq, p, self.board.get(to_sq)))
        return moves

    def is_in_check(self, color: str) -> bool:
        king_pos = self.board.find_king(color)
        if king_pos is None:
            return False
        opponent = 'b' if color == 'w' else 'w'
        return self.board.is_square_attacked(king_pos, opponent)

    def is_move_legal(self, move: Move) -> bool:
        p = self.board.get(move.from_sq)
        if p is None or p.color != move.piece.color or p.kind != move.piece.kind:
            return False
        if move.to_sq not in move.piece.pseudo_legal_moves(self.board, move.from_sq):
            return False
        captured = self.board.make_move_and_return_captured(move)
        in_check = self.is_in_check(move.piece.color)
        self.board.unmake_move(move, captured)
        return not in_check

    def legal_moves_for(self, pos: Pos) -> List[Pos]:
        p = self.board.get(pos)
        if p is None:
            return []
        pseudo = p.pseudo_legal_moves(self.board, pos)
        legal: List[Pos] = []
        for to in pseudo:
            mv = Move(pos, to, p, self.board.get(to))
            if self.is_move_legal(mv):
                legal.append(to)
        return legal


# Pygame UI
class Renderer:
    def __init__(self, surface):
        self.surface = surface
        # try unicode-supporting font
        try:
            self.font = pygame.font.SysFont('segoeuisymbol', 36)
        except Exception:
            self.font = pygame.font.SysFont(None, 36)

    def draw_board(self, board: Board, selected: Optional[Pos], legal_moves: List[Pos]):
        # squares
        for r in range(ROWS):
            for c in range(COLS):
                color = LIGHT if (r + c) % 2 == 0 else DARK
                rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.surface, color, rect)
        # highlight selected
        if selected:
            r, c = selected
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.surface, HIGHLIGHT, rect)
        # highlight legal moves
        for (r, c) in legal_moves:
            center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(self.surface, MOVE_MARK, center, 10)
        # pieces
        for r in range(ROWS):
            for c in range(COLS):
                p = board.get((r, c))
                if p:
                    ch = UNICODE_PIECES.get((p.color, p.kind), p.symbol())
                    text = self.font.render(ch, True, BLACK_TEXT)
                    tx = c * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_width() // 2
                    ty = r * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2
                    self.surface.blit(text, (tx, ty))

    def draw(self, board: Board, selected: Optional[Pos], legal_moves: List[Pos]):
        self.draw_board(board, selected, legal_moves)


class InputHandler:
    @staticmethod
    def pixel_to_square(pos: Tuple[int, int]) -> Pos:
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        col = max(0, min(COLS - 1, col))
        row = max(0, min(ROWS - 1, row))
        return (row, col)


class Game:
    def __init__(self, surface):
        self.surface = surface
        self.clock = pygame.time.Clock()
        self.board = Board()
        self.engine = GameEngine(self.board)
        self.renderer = Renderer(surface)
        self.selected: Optional[Pos] = None
        self.legal_moves: List[Pos] = []
        self.move_history: List[tuple[Move, Optional[Piece]]] = []

    def select_or_move(self, sq: Pos):
        p = self.board.get(sq)
        # select own piece
        if self.selected is None:
            if p and p.color == self.engine.turn:
                self.selected = sq
                self.legal_moves = self.engine.legal_moves_for(sq)
        else:
            # if clicking same square: deselect
            if sq == self.selected:
                self.selected = None
                self.legal_moves = []
                return
            # if clicked on a legal target -> make move
            if sq in self.legal_moves:
                piece = self.board.get(self.selected)
                mv = Move(self.selected, sq, piece, self.board.get(sq))
                # handle promotion simple: if pawn reaches last rank, set promotion to Q
                if piece and piece.kind == 'P' and (sq[0] == 0 or sq[0] == 7):
                    mv.promotion = 'Q'
                captured = self.board.make_move_and_return_captured(mv)
                # if promotion replaced piece, ensure move.piece remains original pawn (for undo)
                self.move_history.append((mv, captured))
                # switch turn
                self.engine.turn = 'b' if self.engine.turn == 'w' else 'w'
                # clear selection
                self.selected = None
                self.legal_moves = []
            else:
                # trying to select another piece of same color
                if p and p.color == self.engine.turn:
                    self.selected = sq
                    self.legal_moves = self.engine.legal_moves_for(sq)
                else:
                    # invalid target: deselect
                    self.selected = None
                    self.legal_moves = []

    def undo(self):
        if not self.move_history:
            return
        mv, captured = self.move_history.pop()
        # revert
        self.board.unmake_move(mv, captured)
        # flip turn back
        self.engine.turn = 'b' if self.engine.turn == 'w' else 'w'
        self.selected = None
        self.legal_moves = []

    def run_once(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                sq = InputHandler.pixel_to_square(event.pos)
                self.select_or_move(sq)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    self.undo()
        # draw
        self.surface.fill((0, 0, 0))
        self.renderer.draw(self.board, self.selected, self.legal_moves)
        pygame.display.flip()
        self.clock.tick(FPS)
        return True


# MAIN
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PYGAME CHESS")
    game = Game(screen)

    running = True
    while running:
        running = game.run_once()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
