"""
Chess (Pygame) — engine + sprites + animation + sounds

This file adds sound effects to the existing chess application.

Sounds expected (place in ./assets/sounds):
- move.wav      -- normal piece move
- capture.wav   -- capture (including en-passant)
- castle.wav    -- castling
- promotion.wav -- promotion
- check.wav     -- move that gives check

All sounds are optional — missing files are silently ignored. Volume can be adjusted in SoundManager.

Run: pip install pygame ; python main.py
"""

import os
import pygame
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Type, Dict

# --- Config
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
FPS = 60
ANIMATION_DURATION = 250  # milliseconds for a single-piece move (tweakable)

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (246, 246, 105)
MOVE_MARK = (200, 50, 50)
BLACK_TEXT = (20, 20, 20)

UNICODE_PIECES = {
    ('w', 'K'): '♔', ('w', 'Q'): '♕', ('w', 'R'): '♖', ('w', 'B'): '♗', ('w', 'N'): '♘', ('w', 'P'): '♙',
    ('b', 'K'): '♚', ('b', 'Q'): '♛', ('b', 'R'): '♜', ('b', 'B'): '♝', ('b', 'N'): '♞', ('b', 'P'): '♟',
}

Pos = Tuple[int, int]


# ----------------- Sound manager -----------------
class SoundManager:
    def __init__(self, assets_dir: str = 'assets/sounds'):
        self.sounds: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self.assets_dir = assets_dir
        self.enabled = True
        try:
            pygame.mixer.init()
        except Exception:
            # audio subsystem unavailable, disable sounds
            self.enabled = False
            return
        self.load('move', 'move.wav')
        self.load('capture', 'capture.wav')
        self.load('castle', 'castle.wav')
        self.load('promotion', 'promotion.wav')
        self.load('check', 'check.wav')
        # default volumes (0.0 - 1.0)
        self.set_volume('move', 0.5)
        self.set_volume('capture', 0.6)
        self.set_volume('castle', 0.6)
        self.set_volume('promotion', 0.7)
        self.set_volume('check', 0.7)

    def load(self, key: str, filename: str):
        if not self.enabled:
            self.sounds[key] = None
            return
        path = os.path.join(self.assets_dir, filename)
        if os.path.exists(path):
            try:
                snd = pygame.mixer.Sound(path)
                self.sounds[key] = snd
            except Exception:
                self.sounds[key] = None
        else:
            self.sounds[key] = None

    def set_volume(self, key: str, vol: float):
        s = self.sounds.get(key)
        if s:
            s.set_volume(max(0.0, min(1.0, vol)))

    def play(self, key: str):
        if not self.enabled:
            return
        s = self.sounds.get(key)
        if s:
            try:
                s.play()
            except Exception:
                pass


# ----------------- Engine: pieces, board, engine -----------------
class Piece:
    def __init__(self, color: str, kind: str):
        assert color in ('w', 'b')
        self.color = color
        self.kind = kind
        self.has_moved = False

    def __repr__(self):
        return f"{self.color}{self.kind}"

    def symbol(self) -> str:
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
                # en-passant capture
                if board.en_passant_target is not None and board.en_passant_target == (tr, tc):
                    moves.append((tr, tc))
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
                nr += dr;
                nc += dc
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
                nr += dr;
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
        # Castling pseudo-legal: basic checks (not checking attacks here)
        if not self.has_moved:
            row = r
            # kingside
            if board.get((row, 7)) is not None and isinstance(board.get((row, 7)), Rook) and not board.get(
                    (row, 7)).has_moved:
                if board.get((row, 5)) is None and board.get((row, 6)) is None:
                    moves.append((row, 6))
            # queenside
            if board.get((row, 0)) is not None and isinstance(board.get((row, 0)), Rook) and not board.get(
                    (row, 0)).has_moved:
                if board.get((row, 1)) is None and board.get((row, 2)) is None and board.get((row, 3)) is None:
                    moves.append((row, 2))
        return moves


_piece_map: Dict[str, Type[Piece]] = {
    'P': Pawn, 'N': Knight, 'B': Bishop, 'R': Rook, 'Q': Queen, 'K': King
}


def piece_from_fen(ch: str) -> Piece:
    color = 'w' if ch.isupper() else 'b'
    kind = ch.upper()
    cls = _piece_map[kind]
    return cls(color)


@dataclass
class Move:
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


class Board:
    ROWS = 8;
    COLS = 8

    def __init__(self):
        self.grid: List[List[Optional[Piece]]] = [[None] * self.COLS for _ in range(self.ROWS)]
        self.en_passant_target: Optional[Pos] = None
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
        # store previous en-passant target for undo
        move.prev_en_passant_target = self.en_passant_target
        move.prev_piece_has_moved = move.piece.has_moved
        # reset en-passant by default
        self.en_passant_target = None

        # Castling
        if move.is_castling:
            # move king
            self.set(move.to_sq, move.piece)
            self.set(move.from_sq, None)
            move.piece.has_moved = True
            # move rook
            rf = move.castling_rook_from
            rt = move.castling_rook_to
            rook = self.get(rf)
            move.prev_rook_has_moved = rook.has_moved if rook else None
            self.set(rt, rook)
            self.set(rf, None)
            if rook:
                rook.has_moved = True
            return None

        # En-passant capture
        if move.is_en_passant and move.en_passant_captured_sq is not None:
            captured = self.get(move.en_passant_captured_sq)
            # move pawn
            self.set(move.to_sq, move.piece)
            self.set(move.from_sq, None)
            # remove captured pawn
            self.set(move.en_passant_captured_sq, None)
            move.piece.has_moved = True
            return captured

        # Normal move / capture
        captured = self.get(move.to_sq)
        self.set(move.to_sq, move.piece)
        self.set(move.from_sq, None)

        # handle pawn double push -> set en-passant target
        if move.piece.kind == 'P':
            if abs(move.to_sq[0] - move.from_sq[0]) == 2:
                # square passed over
                mid_row = (move.to_sq[0] + move.from_sq[0]) // 2
                self.en_passant_target = (mid_row, move.to_sq[1])
        # handle promotion
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
        # revert based on flags
        # revert en-passant target
        self.en_passant_target = move.prev_en_passant_target

        if move.is_castling:
            # move king back
            king = self.get(move.to_sq)
            self.set(move.from_sq, king)
            self.set(move.to_sq, None)
            if king:
                king.has_moved = move.prev_piece_has_moved
            # move rook back
            rf = move.castling_rook_from
            rt = move.castling_rook_to
            rook = self.get(rt)
            self.set(rf, rook)
            self.set(rt, None)
            if rook:
                rook.has_moved = move.prev_rook_has_moved
            return

        if move.is_en_passant and move.en_passant_captured_sq is not None:
            # move pawn back
            pawn = self.get(move.to_sq)
            self.set(move.from_sq, pawn)
            self.set(move.to_sq, None)
            # restore captured pawn
            self.set(move.en_passant_captured_sq, captured)
            if pawn:
                pawn.has_moved = move.prev_piece_has_moved
            return

        # promotion handling: if there was promotion, we replaced pawn with new piece
        if move.promotion and move.piece.kind == 'P':
            # remove promoted piece from to_sq
            self.set(move.to_sq, captured)
            # restore pawn at from_sq
            self.set(move.from_sq, move.piece)
            move.piece.has_moved = move.prev_piece_has_moved
            return

        # normal revert
        self.set(move.from_sq, move.piece)
        self.set(move.to_sq, captured)
        move.piece.has_moved = move.prev_piece_has_moved

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
                nr += dr;
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
                nr += dr;
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
                        rowp += str(empty);
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
                        mv = Move((r, c), to_sq, p, self.board.get(to_sq))
                        # mark castling if king moves two squares horizontally
                        if p.kind == 'K' and abs(to_sq[1] - c) == 2:
                            mv.is_castling = True
                            row = r
                            if to_sq[1] > c:  # kingside
                                mv.castling_rook_from = (row, 7)
                                mv.castling_rook_to = (row, 5)
                            else:
                                mv.castling_rook_from = (row, 0)
                                mv.castling_rook_to = (row, 3)
                        # mark en-passant
                        if p.kind == 'P' and self.board.en_passant_target is not None and to_sq == self.board.en_passant_target:
                            mv.is_en_passant = True
                            ep_row = r
                            ep_col = to_sq[1]
                            mv.en_passant_captured_sq = (r, to_sq[1])
                        moves.append(mv)
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


# ----------------- Animated move helper -----------------
@dataclass
class MovingPiece:
    piece: Piece
    from_sq: Pos
    to_sq: Pos


@dataclass
class AnimatedMove:
    moving: List[MovingPiece]
    move_obj: Move
    start_time_ms: int
    duration_ms: int = ANIMATION_DURATION
    finished: bool = False

    def progress(self, now_ms: int) -> float:
        elapsed = now_ms - self.start_time_ms
        t = min(1.0, max(0.0, elapsed / self.duration_ms))
        if elapsed >= self.duration_ms:
            self.finished = True
        return t


# ----------------- Pygame UI with sprites & animation & sounds -----------------
class Renderer:
    def __init__(self, surface):
        self.surface = surface
        try:
            self.font = pygame.font.SysFont('dejavusans', 36)
        except Exception:
            self.font = pygame.font.SysFont(None, 36)
        self.piece_images: Dict[Tuple[str, str], pygame.Surface] = {}
        self.load_piece_images()

    def load_piece_images(self, assets_dir: str = 'assets'):
        for color in ('w', 'b'):
            for kind in ('K', 'Q', 'R', 'B', 'N', 'P'):
                fname = os.path.join(assets_dir, f"{color}{kind}.png")
                if os.path.exists(fname):
                    try:
                        img = pygame.image.load(fname).convert_alpha()
                        img = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                        self.piece_images[(color, kind)] = img
                    except Exception:
                        pass

    def draw_board(self, board: Board, selected: Optional[Pos], legal_moves: List[Pos],
                   animated_move: Optional[AnimatedMove]):
        for r in range(ROWS):
            for c in range(COLS):
                color = LIGHT if (r + c) % 2 == 0 else DARK
                rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.surface, color, rect)
        if selected:
            r, c = selected
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.surface, HIGHLIGHT, rect)
        for (r, c) in legal_moves:
            center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(self.surface, MOVE_MARK, center, 10)
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
            # try to draw piece sprite inside popup if available
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
        self.draw_board(board, selected, legal_moves, animated_move)
        if promotion_center and promotion_color:
            return self.draw_promotion_popup(promotion_color, promotion_center)
        return None


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
        # pending promotion
        self.pending_promotion_move: Optional[Move] = None
        self.promotion_rects: Optional[List[Tuple[pygame.Rect, str]]] = None
        # animation state
        self.animated_move: Optional[AnimatedMove] = None
        # sounds
        self.sounds = SoundManager()

    def start_animation_for_move(self, mv: Move):
        moving = []
        if mv.is_castling and mv.castling_rook_from and mv.castling_rook_to:
            king = mv.piece
            rook = self.board.get(mv.castling_rook_from)
            moving.append(MovingPiece(king, mv.from_sq, mv.to_sq))
            if rook is not None:
                moving.append(MovingPiece(rook, mv.castling_rook_from, mv.castling_rook_to))
        elif mv.is_en_passant and mv.en_passant_captured_sq is not None:
            moving.append(MovingPiece(mv.piece, mv.from_sq, mv.to_sq))
        else:
            moving.append(MovingPiece(mv.piece, mv.from_sq, mv.to_sq))
        anim = AnimatedMove(moving, mv, pygame.time.get_ticks(), ANIMATION_DURATION)
        self.animated_move = anim

    def finalize_animated_move(self):
        if not self.animated_move:
            return
        mv = self.animated_move.move_obj
        captured = self.board.make_move_and_return_captured(mv)
        # play appropriate sounds based on result
        if mv.is_castling:
            # castle sound already played on start; also play a short confirmation
            self.sounds.play('castle')
        elif mv.promotion:
            self.sounds.play('promotion')
        elif captured is not None:
            self.sounds.play('capture')
        else:
            self.sounds.play('move')
        # check if opponent in check after move
        opponent = 'b' if mv.piece.color == 'w' else 'w'
        if self.engine.is_in_check(opponent):
            self.sounds.play('check')
        self.move_history.append((mv, captured))
        self.engine.turn = 'b' if self.engine.turn == 'w' else 'w'
        self.animated_move = None
        self.selected = None
        self.legal_moves = []

    def select_or_move(self, sq: Pos):
        if self.pending_promotion_move is not None:
            return
        if self.animated_move is not None:
            return
        p = self.board.get(sq)
        if self.selected is None:
            if p and p.color == self.engine.turn:
                self.selected = sq
                self.legal_moves = self.engine.legal_moves_for(sq)
        else:
            if sq == self.selected:
                self.selected = None
                self.legal_moves = []
                return
            if sq in self.legal_moves:
                piece = self.board.get(self.selected)
                mv = Move(self.selected, sq, piece, self.board.get(sq))
                if piece and piece.kind == 'K' and abs(sq[1] - self.selected[1]) == 2:
                    mv.is_castling = True
                    row = self.selected[0]
                    if sq[1] > self.selected[1]:
                        mv.castling_rook_from = (row, 7);
                        mv.castling_rook_to = (row, 5)
                    else:
                        mv.castling_rook_from = (row, 0);
                        mv.castling_rook_to = (row, 3)
                if piece and piece.kind == 'P' and self.board.en_passant_target is not None and sq == self.board.en_passant_target:
                    mv.is_en_passant = True
                    mv.en_passant_captured_sq = (self.selected[0], sq[1])
                if piece and piece.kind == 'P' and (sq[0] == 0 or sq[0] == 7):
                    self.pending_promotion_move = mv
                    px = sq[1] * SQUARE_SIZE + SQUARE_SIZE // 2
                    py = sq[0] * SQUARE_SIZE + SQUARE_SIZE // 2
                    self.promotion_center = (px, py)
                    return
                self.start_animation_for_move(mv)
            else:
                if p and p.color == self.engine.turn:
                    self.selected = sq
                    self.legal_moves = self.engine.legal_moves_for(sq)
                else:
                    self.selected = None
                    self.legal_moves = []

    def handle_promotion_click(self, pos):
        if not self.pending_promotion_move:
            return
        if not self.promotion_rects:
            return
        for rect, choice in self.promotion_rects:
            if rect.collidepoint(pos):
                mv = self.pending_promotion_move
                mv.promotion = choice
                self.start_animation_for_move(mv)
                self.pending_promotion_move = None
                self.promotion_rects = None
                self.selected = None
                self.legal_moves = []
                return
        self.pending_promotion_move = None
        self.promotion_rects = None

    def undo(self):
        if not self.move_history or self.animated_move is not None:
            return
        mv, captured = self.move_history.pop()
        self.board.unmake_move(mv, captured)
        self.engine.turn = 'b' if self.engine.turn == 'w' else 'w'
        self.selected = None
        self.legal_moves = []
        self.pending_promotion_move = None
        self.promotion_rects = None

    def run_once(self) -> bool:
        now_ms = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.pending_promotion_move is not None:
                    self.handle_promotion_click(event.pos)
                else:
                    sq = InputHandler.pixel_to_square(event.pos)
                    self.select_or_move(sq)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    self.undo()
        if self.animated_move is not None:
            if self.animated_move.finished:
                self.finalize_animated_move()
        self.surface.fill((0, 0, 0))
        rects = None
        if self.pending_promotion_move is not None:
            rects = self.renderer.draw(self.board, self.selected, self.legal_moves, self.animated_move,
                                       self.promotion_center, self.pending_promotion_move.piece.color)
            self.promotion_rects = rects
        else:
            self.renderer.draw(self.board, self.selected, self.legal_moves, self.animated_move)
        pygame.display.flip()
        self.clock.tick(FPS)
        return True


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess - Sprites, Animation & Sounds")
    game = Game(screen)

    running = True
    while running:
        running = game.run_once()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
