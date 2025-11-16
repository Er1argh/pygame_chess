"""Простой менеджер звуковых эффектов на базе pygame.mixer с грациозным деградантом.

SoundManager загружает набор именованных звуковых эффектов из директории
(по умолчанию 'assets/sounds') и предоставляет методы для управления громкостью
и воспроизведения. Если инициализация аудио подсистемы не удалась, менеджер
переходит в «беззвучный» режим — все вызовы play будут игнорироваться, а
попытки загрузки запишут None в словарь, что упрощает обработку в вызывающем коде.

Пример:
    sm = SoundManager()
    sm.play('move')  # если аудио доступно — проиграет звук, иначе — тихо ничего не сделает
"""

import os
from typing import Dict, Optional

import pygame


class SoundManager:
    """Менеджер именованных звуковых эффектов.

    Менеджер пытается инициализировать `pygame.mixer` при создании. После
    успешной инициализации он загружает заранее определённый набор файлов
    (move.wav, capture.wav и т.д.) из `assets_dir`. В случае ошибки инициализации
    аудио (например, на системе без звука) менеджер помечается как отключённый
    и далее ведёт себя как no-op: загрузка помечает ключи со значением None,
    метод `play` ничего не выполняет.

    Args:
        assets_dir: Папка, откуда загружать звуковые файлы (по умолчанию 'assets/sounds').

    Attributes:
        sounds: Словарь {key: pygame.mixer.Sound | None} с загруженными эффектами.
        assets_dir: путь к папке с ресурсами.
        enabled: True, если pygame.mixer успешно инициализирован, иначе False.
    """

    def __init__(self, assets_dir: str = 'assets/sounds'):
        self.sounds: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self.assets_dir = assets_dir
        self.enabled = True

        try:
            pygame.mixer.init()
        except Exception:
            self.enabled = False
            return

        self.load('move', 'move.wav')
        self.load('capture', 'capture.wav')
        self.load('castle', 'castle.wav')
        self.load('promotion', 'promotion.wav')
        self.load('check', 'check.wav')
        self.set_volume('move', 0.5)
        self.set_volume('capture', 0.6)
        self.set_volume('castle', 0.6)
        self.set_volume('promotion', 0.7)
        self.set_volume('check', 0.7)

    def load(self, key: str, filename: str):
        """Загрузить звуковой файл и сохранить его под ключом `key`.

        Если менеджер выключен (enabled == False) или файл не найден /
        неудачно загружен — в словаре будет значение None.

        Args:
            key: ключ, по которому звук будет доступен (например, 'move').
            filename: имя файла внутри assets_dir (например, 'move.wav').
        """
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
        """Установить громкость для звука `key`.

        Громкость автоматически ограничивается диапазоном [0.0, 1.0]. Если
        звук не загружен или менеджер отключён — метод тихо ничего не делает.

        Args:
            key: ключ звука.
            vol: желаемая громкость (float).
        """
        s = self.sounds.get(key)

        if s:
            s.set_volume(max(0.0, min(1.0, vol)))

    def play(self, key: str):
        """Проиграть звук, связанный с `key`, если он доступен и звук включён.

        Если менеджер находится в беззвучном режиме (enabled == False) — вызов
        игнорируется. Если звук отсутствует (None), метод также ничего не делает.

        Args:
            key: ключ звукового эффекта (например, 'move', 'capture').
        """
        if not self.enabled:
            return

        s = self.sounds.get(key)

        if s:
            try:
                s.play()
            except Exception:
                pass
