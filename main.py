import pygame
import sys
from typing import Optional, Tuple, List, Dict

# CONSTANTS
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
FPS = 60

WHITE = (245, 245, 220)
BROWN = (118, 150, 86)
HIGHLIGHT = (186, 202, 68)
BLACK = (20, 20, 20)


# MODEL CLASSES
class Piece:
    """"
    Базовый класс фигуры.
    """

    def __init__(self, color: str, kind: str):
        self.color = color  # 'w' | 'b'
        self.kind = kind  # 'K' | 'Q' | 'R' | 'B' | 'N' | 'P'

    def __repr__(self):
        # Цвет и тип фигуры (wK, например)
        return f"{self.color}{self.kind}"

    def symbol(self) -> str:
        # Текстовая метка для отрисовки (только тип фигуры без цвета)
        return {'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P'}[self.kind]


class Pawn(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'P')


class Rook(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'R')


class Knight(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'N')


class Bishop(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'B')


class Queen(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'Q')


class King(Piece):
    def __init__(self, color: str):
        super().__init__(color, 'K')


class PieceFactory:
    # Создает фигуру из буквенных обозначений (большая буква - белый цвет, маленькая - черный)
    @staticmethod
    def from_fen_char(ch: str) -> Piece:
        color = 'w' if ch.isupper() else 'b'
        kind = ch.upper()
        mapping = {'P': Pawn, 'R': Rook, 'N': Knight, 'B': Bishop, 'Q': Queen, 'K': King}
        cls = mapping[kind]
        return cls(color)


# Move container
class Move:
    def __init__(self, from_sq: Tuple[int, int], to_sq: Tuple[int, int], piece: Piece,
                 captured: Optional[Piece] = None):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.piece = piece
        self.captured = captured

    def __repr__(self):
        return f"Move({self.from_sq} -> {self.to_sq}, {self.piece}, cap={self.captured})"


# Board model
class Board:
    def __init__(self):
        # grid[row][col]; row 0 - top (8th rank), row 7 - bottom (1st rank)
        self.grid: List[List[Optional[Piece]]] = [[None] * COLS for _ in range(ROWS)]
        self.setup_start_position()

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
        for r in range(ROWS):
            row = start_fen[r]
            for c, ch in enumerate(row):
                if ch == '.':
                    self.grid[r][c] = None
                else:
                    self.grid[r][c] = PieceFactory.from_fen_char(ch)

    def get(self, pos: Tuple[int, int]) -> Optional[Piece]:
        r, c = pos
        return self.grid[r][c]

    def set(self, pos: Tuple[int, int], piece: Optional[Piece]):
        r, c = pos
        self.grid[r][c] = piece

    def move_piece(self, move: Move):
        # no rule checking here
        self.set(move.to_sq, move.piece)
        self.set(move.from_sq, None)


# VIEW
class Renderer:
    def __init__(self, surface):
        self.surface = surface
        self.font = pygame.font.SysFont('dejavusans', 36)

    def draw_board(self, board: Board, selected: Optional[Tuple[int, int]]):
        # draw squares
        for r in range(ROWS):
            for c in range(COLS):
                color = WHITE if (r + c) % 2 == 0 else BROWN
                rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.surface, color, rect)

        # highlight selected
        if selected:
            r, c = selected
            rect = pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.surface, HIGHLIGHT, rect)

        # draw pieces as text
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.get((r, c))
                if piece:
                    text = self.font.render(piece.symbol(), True, BLACK if piece.color == 'w' else (0, 0, 0))
                    # center text
                    tx = c * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_width() // 2
                    ty = r * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2
                    self.surface.blit(text, (tx, ty))

    def draw(self, board: Board, selected: Optional[Tuple[int, int]]):
        self.draw_board(board, selected)


# INPUT HANDLING & CONTROLLER
class InputHandler:
    """
    Обрабатывает клики и конвертирует их в координаты доски.
    """

    @staticmethod
    def pixel_to_square(pos: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        # clamp
        col = max(0, min(COLS - 1, col))
        row = max(0, min(ROWS - 1, row))
        return (row, col)


class Game:
    def __init__(self, surface):
        self.surface = surface
        self.clock = pygame.time.Clock()
        self.board = Board()
        self.renderer = Renderer(surface)
        self.selected: Optional[Tuple[int, int]] = None
        self.turn = 'w'
        self.move_history: List[Move] = []

    def handle_click(self, pos: Tuple[int, int]):
        sq = InputHandler.pixel_to_square(pos)
        piece = self.board.get(sq)
        if self.selected is None:
            # select piece if any and if it belongs to player to move
            if piece and piece.color == self.turn:
                self.selected = sq
        else:
            # attempt to move selected -> sq (no rule checking for now)
            from_sq = self.selected
            moving_piece = self.board.get(from_sq)
            if moving_piece:
                captured = self.board.get(sq)
                move = Move(from_sq, sq, moving_piece, captured)
                self.board.move_piece(move)
                self.move_history.append(move)
                # switch turn
                self.turn = 'b' if self.turn == 'w' else 'w'
            self.selected = None

    def run_once(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    self.undo_last()
        # draw
        self.surface.fill((0, 0, 0))
        self.renderer.draw(self.board, self.selected)
        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def undo_last(self):
        if not self.move_history:
            return
        last = self.move_history.pop()
        # revert
        self.board.set(last.from_sq, last.piece)
        self.board.set(last.to_sq, last.captured)
        # switch turn back
        self.turn = 'b' if self.turn == 'w' else 'w'


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
