import json
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sign_mapper import SignMapper


class SignToAudioEngine:
    """Webcam sign recognizer (template-based) with optional text-to-speech and sign-video echo."""

    def __init__(
        self,
        templates_path: str,
        camera_index: int = 0,
        min_confidence: float = 0.6,
        match_threshold: float = 0.16,
        stable_frames: int = 8,
        speak_enabled: bool = True,
        sign_dir: str = "sign_videos",
    ):
        self.templates_path = templates_path
        self.camera_index = camera_index
        self.min_confidence = min_confidence
        self.match_threshold = match_threshold
        self.stable_frames = stable_frames
        self.speak_enabled = speak_enabled

        self._tts_queue: "queue.Queue[str]" = queue.Queue()
        self._tts_thread: Optional[threading.Thread] = None
        self._stop_tts = threading.Event()

        self._mp = None
        self._hands = None
        self._templates = self._load_templates(templates_path)

        self._last_candidate = ""
        self._candidate_count = 0
        self._last_committed = ""
        self._last_commit_time = 0.0
        self._recognized_phrase: List[str] = []

        self._mapper = SignMapper(sign_dir)
        self._video_queue: "queue.Queue[str]" = queue.Queue()
        self._current_cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        if not self._templates:
            print(f"No templates loaded from: {self.templates_path}")
            print("Create templates first. Expected format: {\"label\": [[42 floats], ...]}")

        try:
            import mediapipe as mp
        except Exception as exc:
            print(f"Startup error: sign-to-audio mode requires 'mediapipe': {exc}")
            return

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        drawer = mp.solutions.drawing_utils

        if self.speak_enabled:
            self._start_tts_worker()

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Startup error: cannot open camera index {self.camera_index}")
            self._shutdown()
            return

        window_name = "Sign to Audio/Text"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Sign-to-audio mode started. Press 'q' to quit, 'c' to clear transcript.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._hands.process(rgb)

                candidate, distance = "", None
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    drawer.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self._mp.solutions.hands.HAND_CONNECTIONS,
                    )
                    vec = self._extract_vector(hand_landmarks)
                    candidate, distance = self._classify(vec)

                committed = self._update_stability(candidate)
                if committed:
                    self._recognized_phrase.append(committed)
                    if self.speak_enabled:
                        self._tts_queue.put(committed)

                    # Optional sign-video echo for recognized token.
                    video_paths, _ = self._mapper.map_tokens([committed])
                    for path in video_paths:
                        self._video_queue.put(path)

                sign_frame = self._next_sign_frame()
                if sign_frame is not None:
                    frame = self._paste_sign_overlay(frame, sign_frame)

                self._draw_hud(frame, candidate, distance)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self._recognized_phrase = []
                    self._last_committed = ""
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._shutdown()

    def _load_templates(self, path: str) -> Dict[str, np.ndarray]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return {}

        templates: Dict[str, np.ndarray] = {}
        if not isinstance(raw, dict):
            return templates

        for label, samples in raw.items():
            if not isinstance(label, str) or not isinstance(samples, list) or not samples:
                continue

            valid_rows = []
            for row in samples:
                if not isinstance(row, list):
                    continue
                arr = np.asarray(row, dtype=np.float32)
                if arr.ndim == 1 and arr.size == 42:
                    valid_rows.append(arr)

            if valid_rows:
                templates[label.lower().strip()] = np.stack(valid_rows)

        return templates

    def _extract_vector(self, hand_landmarks) -> np.ndarray:
        points = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)

        origin = points[0]
        centered = points - origin
        max_norm = np.linalg.norm(centered, axis=1).max()
        if max_norm > 1e-6:
            centered /= max_norm

        return centered.reshape(-1)

    def _classify(self, vector: np.ndarray) -> Tuple[str, Optional[float]]:
        if not self._templates:
            return "", None

        best_label = ""
        best_distance = 1e9

        for label, samples in self._templates.items():
            dists = np.linalg.norm(samples - vector, axis=1)
            score = float(np.min(dists))
            if score < best_distance:
                best_distance = score
                best_label = label

        if best_distance <= self.match_threshold:
            return best_label, best_distance
        return "", best_distance

    def _update_stability(self, candidate: str) -> str:
        if not candidate:
            self._last_candidate = ""
            self._candidate_count = 0
            return ""

        if candidate == self._last_candidate:
            self._candidate_count += 1
        else:
            self._last_candidate = candidate
            self._candidate_count = 1

        if self._candidate_count < self.stable_frames:
            return ""

        now = time.time()
        if candidate == self._last_committed and (now - self._last_commit_time) < 1.2:
            return ""

        self._last_committed = candidate
        self._last_commit_time = now
        self._candidate_count = 0
        self._last_candidate = ""
        return candidate

    def _draw_hud(self, frame: np.ndarray, candidate: str, distance: Optional[float]) -> None:
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)

        phrase = " ".join(self._recognized_phrase[-8:])
        if not phrase:
            phrase = "Show a known sign to camera..."

        status = "No match"
        if candidate:
            status = f"Candidate: {candidate}"
        elif distance is not None:
            status = f"No confident match (dist={distance:.3f})"

        cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Output: {phrase}", (12, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "q: quit | c: clear", (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    def _next_sign_frame(self) -> Optional[np.ndarray]:
        while True:
            if self._current_cap is None:
                self._current_cap = self._open_next_video()
                if self._current_cap is None:
                    return None

            ok, frame = self._current_cap.read()
            if not ok:
                self._release_current_cap()
                continue

            return cv2.resize(frame, (240, 180), interpolation=cv2.INTER_AREA)

    def _open_next_video(self) -> Optional[cv2.VideoCapture]:
        while True:
            try:
                path = self._video_queue.get_nowait()
            except queue.Empty:
                return None

            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                return cap
            cap.release()

    def _release_current_cap(self) -> None:
        if self._current_cap is not None:
            self._current_cap.release()
            self._current_cap = None

    def _paste_sign_overlay(self, canvas: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        h, w = canvas.shape[:2]
        oh, ow = overlay.shape[:2]
        x1 = w - ow - 16
        y1 = 46
        x2 = x1 + ow
        y2 = y1 + oh

        cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (220, 220, 220), 2)
        canvas[y1:y2, x1:x2] = overlay
        return canvas

    def _start_tts_worker(self) -> None:
        self._stop_tts.clear()
        self._tts_thread = threading.Thread(target=self._tts_loop, name="sign-tts", daemon=True)
        self._tts_thread.start()

    def _tts_loop(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        except Exception as exc:
            print(f"TTS disabled: cannot initialize pyttsx3 ({exc})")
            return

        while not self._stop_tts.is_set():
            try:
                text = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                continue

    def _shutdown(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

        self._release_current_cap()
        self._stop_tts.set()
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=1.5)
        self._tts_thread = None
