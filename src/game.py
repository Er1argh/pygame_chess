"""Модуль реализации игрового цикла и управления состоянием игры.

Содержит класс `Game`, управляющий интерфейсом, анимациями, звуками
и взаимодействием с игровым движком (`GameEngine`).
Функция `main` инициализирует Pygame и запускает цикл.
"""

import sys
from typing import Optional, List, Tuple

import pygame

from src.anim import AnimatedMove, MovingPiece
from src.config import Pos, ANIMATION_DURATION, SQUARE_SIZE, FPS, WIDTH, HEIGHT
from src.engine import Board, GameEngine, Move, Piece
from src.input_handler import InputHandler
from src.renderer import Renderer
from src.sounds import SoundManager


def show_centered_message(screen, message: str,
                          font_name: str = "Calibri",
                          base_size: int = None,
                          text_color=(245, 245, 245),
                          overlay_color=(10, 10, 10, 180)):
    """
    Draws a semi-transparent overlay and centers the message on screen.
    - text_color is RGB tuple
    - overlay_color is RGBA for semi-transparent background
    - base_size: if None, computed from screen width
    """
    w, h = screen.get_size()
    # adaptive font size
    if base_size is None:
        base_size = max(28, min(72, w // 12))
    try:
        font = pygame.font.SysFont(font_name, base_size, True)
    except Exception:
        font = pygame.font.SysFont(None, base_size, True)

    # overlay
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill(overlay_color)  # RGBA
    screen.blit(overlay, (0, 0))

    # render shadow then text for better contrast
    text_surf = font.render(message, True, text_color)
    shadow_surf = font.render(message, True, (0, 0, 0))

    x = w // 2 - text_surf.get_width() // 2
    y = h // 2 - text_surf.get_height() // 2

    # draw shadow slightly offset
    screen.blit(shadow_surf, (x + 3, y + 3))
    screen.blit(text_surf, (x, y))

    # optionally update display here (or do it in main loop)
    pygame.display.update()


class Game:
    """Контроллер игры: обработка событий, управление анимацией и отрисовкой.

    Объект `Game` связывает рендерер, игровой движок и аудио-менеджер.
    Он принимает Pygame-серфейс (surface) и реализует один шаг игрового цикла
    через метод `run_once`.

    Attributes:
        surface: Pygame surface для рисования.
        clock: Pygame clock для фиксации FPS.
        board: Объект `Board` с текущей позицией.
        engine: Объект `GameEngine` для вычисления легальных ходов и проверки шаха.
        renderer: Объект `Renderer` для отрисовки доски и фигур.
        selected: Выбранная клетка (row, col) или None.
        legal_moves: Список легальных ходов для выбранной фигуры.
        move_history: История ходов (Move, captured_piece).
        pending_promotion_move: Если пешка достигла последней горизонтали — отложенный ход с промоцией.
        promotion_rects: Прямоугольники для клика в окне выбора промоции.
        animated_move: Текущая анимация перемещения или None.
        sounds: Менеджер звуков.
        promotion_center: экранная позиция центра popup-а выбора промоции или None.
    """

    def __init__(self, surface):
        """Создать контроллер игры и инициализировать все подсистемы.

        Args:
            surface: Pygame surface, куда будет рендериться игра.
        """
        self.surface = surface
        self.clock = pygame.time.Clock()
        self.board = Board()
        self.engine = GameEngine(self.board)
        self.renderer = Renderer(surface)
        self.selected: Optional[Pos] = None
        self.legal_moves: List[Pos] = []
        self.move_history: List[tuple[Move, Optional[Piece]]] = []
        self.pending_promotion_move: Optional[Move] = None
        self.promotion_rects: Optional[List[Tuple[pygame.Rect, str]]] = None
        self.animated_move: Optional[AnimatedMove] = None
        self.sounds = SoundManager()
        self.promotion_center: Optional[Pos] = None
        self.half_counter_moves = 0

    def start_animation_for_move(self, mv: Move):
        """Подготовить и начать анимацию для хода.

        Создаёт объекты `MovingPiece` для всех фигур, участвующих в анимации
        (король и ладья при рокировке, пешка при взятии на проходе и т.д.)
        и сохраняет `AnimatedMove` в `self.animated_move`.

        Args:
            mv: Объект `Move`, который нужно анимировать.
        """
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
        """Завершить текущую анимацию и применить соответствующий ход.

        Метод применяет ход к доске (`Board.make_move_and_return_captured`),
        проигрывает подходящие звуки, обновляет историю ходов и переключает ход.
        Если никакой анимации нет — метод ничего не делает.
        """
        if not self.animated_move:
            return

        mv = self.animated_move.move_obj
        captured = self.board.make_move_and_return_captured(mv)

        if mv.is_castling:
            self.sounds.play('castle')
        elif mv.promotion:
            self.sounds.play('promotion')
        elif captured is not None:
            self.sounds.play('capture')
        else:
            self.sounds.play('move')

        opponent = 'b' if mv.piece.color == 'w' else 'w'

        if self.engine.is_in_check(opponent):
            self.sounds.play('check')

        self.move_history.append((mv, captured))
        self.engine.turn = 'b' if self.engine.turn == 'w' else 'w'
        self.animated_move = None
        self.selected = None
        self.legal_moves = []

    def select_or_move(self, sq: Pos):
        """Выбрать фигуру или попытаться выполнить ход из/в клетку `sq`.

        Алгоритм:
        - Если ожидается выбор промоции — игнорируется.
        - Если идёт анимация — игнорируется.
        - Если ничего не выбрано и в `sq` стоит фигура текущего цвета — выбрать её.
        - Если уже выбранная клетка — снять выбор.
        - Если клик соответствует легальному ходу — сформировать `Move` и начать анимацию.
        - Обработаны случаи рокировки, взятия на проходе и отложенной промоции.

        Args:
            sq: Целевая клетка в координатах (row, col).
        """
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
                        mv.castling_rook_from = (row, 7)
                        mv.castling_rook_to = (row, 5)
                    else:
                        mv.castling_rook_from = (row, 0)
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
                if piece and piece.kind != 'P' and mv.captured is None:
                    self.half_counter_moves += 1
                else:
                    self.half_counter_moves = 0
            else:
                if p and p.color == self.engine.turn:
                    self.selected = sq
                    self.legal_moves = self.engine.legal_moves_for(sq)
                else:
                    self.selected = None
                    self.legal_moves = []

    def handle_promotion_click(self, pos):
        """Обработать клик по окну выбора промоции.

        Если клик попадает в один из прямоугольников `promotion_rects`,
        назначается тип промоции и ход анимируется. Если клик вне области —
        отменяем отложенную промоцию.

        Args:
            pos: Координаты пикселя (x, y) клика мыши.
        """
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
        """Отменить последний ход (undo).

        Восстанавливает состояние доски из `move_history` и переключает очередь хода.
        Ничего не делает, если есть текущая анимация или история пуста.
        """
        if not self.move_history or self.animated_move is not None:
            return

        mv, captured = self.move_history.pop()
        self.board.unmake_move(mv, captured)
        self.engine.turn = 'b' if self.engine.turn == 'w' else 'w'
        self.selected = None
        self.legal_moves = []
        self.pending_promotion_move = None
        self.promotion_rects = None

    def run_once(self, screen) -> bool:
        """Выполнить одну итерацию игрового цикла.

        Обрабатывает события Pygame (клик мышью, клавиши), обновляет анимации,
        перерисовывает экран и ждёт следующий кадр.

        Returns:
            bool: False если нужно завершить игру (например, было событие QUIT),
                  True для продолжения игрового цикла.
        """
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

        if self.pending_promotion_move is not None:
            rects = self.renderer.draw(self.board, self.selected, self.legal_moves, self.animated_move,
                                       self.promotion_center, self.pending_promotion_move.piece.color)
            self.promotion_rects = rects
        else:
            self.renderer.draw(self.board, self.selected, self.legal_moves, self.animated_move)

        leg_moves_for_black = []
        for move in self.engine.all_pseudo_legal_moves_for('b'):
            if self.engine.is_move_legal(move):
                leg_moves_for_black.append(move)

        leg_moves_for_white = []
        for move in self.engine.all_pseudo_legal_moves_for('w'):
            if self.engine.is_move_legal(move):
                leg_moves_for_white.append(move)

        # Checkmate for White (dark red-ish overlay, white text)
        if self.engine.is_in_check('w') and leg_moves_for_white == []:
            show_centered_message(screen, "Checkmate for White",
                                  text_color=(245, 245, 245),
                                  overlay_color=(60, 10, 10, 200))

        # Checkmate for Black (dark blue-ish overlay, white text)
        if self.engine.is_in_check('b') and leg_moves_for_black == []:
            show_centered_message(screen, "Checkmate for Black",
                                  text_color=(245, 245, 245),
                                  overlay_color=(10, 30, 60, 200))

        # Stalemate (muted golden text on dark overlay)
        if (not self.engine.is_in_check('w') and not self.engine.is_in_check('b')) and \
                (leg_moves_for_white == [] or leg_moves_for_black == []):
            show_centered_message(screen, "Stalemate",
                                  text_color=(240, 210, 120),
                                  overlay_color=(20, 20, 20, 190))

        # 50-move / half-move draw (neutral gray on dark overlay)
        if self.half_counter_moves >= 100:
            show_centered_message(screen, "Draw",
                                  text_color=(220, 220, 220),
                                  overlay_color=(15, 15, 15, 200))

        # if self.engine.is_in_check('w') and leg_moves_for_white == []:
        #     font = pygame.font.SysFont('Calibri', 60, True)
        #     text = font.render('Checkmate for White', True, (255, 255, 255))
        #     screen.blit(text, (100, 200))
        #     pygame.display.update()
        #
        # if self.engine.is_in_check('b') and leg_moves_for_black == []:
        #     font = pygame.font.SysFont('Calibri', 60, True)
        #     text = font.render('Checkmate for Black', True, (255, 255, 255))
        #     screen.blit(text, (100, 200))
        #     pygame.display.update()
        #
        # if not self.engine.is_in_check('w') and not self.engine.is_in_check('b'):
        #     if leg_moves_for_white == [] or leg_moves_for_black == []:
        #         font = pygame.font.SysFont('Calibri', 60, True)
        #         text = font.render('Stalemate', True, (255, 255, 255))
        #         screen.blit(text, (100, 200))
        #         pygame.display.update()
        #
        # if self.half_counter_moves >= 100:
        #     screen.fill((0,0,0))
        #     font = pygame.font.SysFont('Calibri', 60, True)
        #     text = font.render('Draw', True, (255, 255, 255))
        #     screen.blit(text, (100, 200))
        #     pygame.display.update()

        pygame.display.flip()
        self.clock.tick(FPS)
        return True


def main():
    """Запустить игровой цикл.

    Инициализирует Pygame, создаёт окно и объект `Game`, затем выполняет цикл
    `while running` до получения события выхода. По завершении корректно завершает Pygame.

    Эта функция предназначена для запуска как скрипт (точка входа).
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PYGAME CHESS")
    game = Game(screen)

    running = True

    while running:
        running = game.run_once(screen)

    pygame.quit()
    sys.exit()
