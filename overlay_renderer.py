import os
import queue
import threading
from typing import List, Optional

import cv2
import numpy as np


class OverlayRenderer:
    """Renders main canvas + subtitle + bottom-right sign-video overlay."""

    def __init__(self, width: int = 1280, height: int = 720, overlay_w: int = 360, overlay_h: int = 260):
        self.width = width
        self.height = height
        self.overlay_w = overlay_w
        self.overlay_h = overlay_h

        self._subtitle_lock = threading.Lock()
        self._subtitle_text: str = ""

        self._video_queue: "queue.Queue[str]" = queue.Queue()
        self._current_cap: Optional[cv2.VideoCapture] = None
        self._window_name = "Speech to Sign Interpreter"

    def set_subtitle(self, text: str) -> None:
        with self._subtitle_lock:
            self._subtitle_text = text

    def enqueue_videos(self, video_paths: List[str]) -> None:
        for path in video_paths:
            self._video_queue.put(path)

    def clear_video_queue(self) -> None:
        while not self._video_queue.empty():
            try:
                self._video_queue.get_nowait()
            except queue.Empty:
                break

    def run(self, stop_event: threading.Event) -> bool:
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, self.width, self.height)

        while not stop_event.is_set():
            frame = self._build_background()
            sign_frame = self._next_sign_frame()
            if sign_frame is not None:
                frame = self._paste_overlay(frame, sign_frame)
            else:
                frame = self._draw_overlay_placeholder(frame)

            subtitle = self._get_subtitle()
            frame = self._draw_subtitle(frame, subtitle)

            cv2.imshow(self._window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break

        self._release_current_cap()
        cv2.destroyAllWindows()
        return True

    def _build_background(self) -> np.ndarray:
        # Simple neutral studio-like gradient background.
        y = np.linspace(0, 1, self.height).reshape(-1, 1)
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bg[:, :, 0] = (35 + 25 * (1 - y)).astype(np.uint8)
        bg[:, :, 1] = (45 + 35 * (1 - y)).astype(np.uint8)
        bg[:, :, 2] = (55 + 45 * (1 - y)).astype(np.uint8)
        return bg

    def _next_sign_frame(self) -> Optional[np.ndarray]:
        while True:
            if self._current_cap is None:
                self._current_cap = self._open_next_video()
                if self._current_cap is None:
                    return None

            ret, frame = self._current_cap.read()
            if not ret:
                self._release_current_cap()
                continue

            return cv2.resize(frame, (self.overlay_w, self.overlay_h), interpolation=cv2.INTER_AREA)

    def _open_next_video(self) -> Optional[cv2.VideoCapture]:
        while True:
            try:
                next_path = self._video_queue.get_nowait()
            except queue.Empty:
                return None

            if not os.path.isfile(next_path):
                continue

            cap = cv2.VideoCapture(next_path)
            if cap.isOpened():
                return cap
            cap.release()

    def _release_current_cap(self) -> None:
        if self._current_cap is not None:
            self._current_cap.release()
            self._current_cap = None

    def _paste_overlay(self, canvas: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        x1 = self.width - self.overlay_w - 24
        y1 = self.height - self.overlay_h - 90
        x2 = x1 + self.overlay_w
        y2 = y1 + self.overlay_h

        cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (230, 230, 230), 2)
        canvas[y1:y2, x1:x2] = overlay
        return canvas

    def _draw_overlay_placeholder(self, canvas: np.ndarray) -> np.ndarray:
        x1 = self.width - self.overlay_w - 24
        y1 = self.height - self.overlay_h - 90
        x2 = x1 + self.overlay_w
        y2 = y1 + self.overlay_h

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (200, 200, 200), 2)
        cv2.putText(canvas, "Waiting for sign video...", (x1 + 18, y1 + self.overlay_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
        return canvas

    def _draw_subtitle(self, frame: np.ndarray, subtitle: str) -> np.ndarray:
        box_h = 74
        x1, y1 = 0, self.height - box_h
        x2, y2 = self.width, self.height

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        text = subtitle if subtitle else "Listening..."
        max_chars = 90
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."

        cv2.putText(frame, text, (20, self.height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _get_subtitle(self) -> str:
        with self._subtitle_lock:
            return self._subtitle_text
