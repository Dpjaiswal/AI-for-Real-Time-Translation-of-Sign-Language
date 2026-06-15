from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np


# URL for the pre-built MediaPipe hand landmarker task bundle
_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
# Cached locally so we only download once
_DEFAULT_TASK_PATH = Path(__file__).resolve().parent.parent.parent / "runtime" / "hand_landmarker.task"


@dataclass
class GesturePrediction:
    candidate: str = ""
    accepted: bool = False
    confidence: float = 0.0
    distance: Optional[float] = None
    has_hand: bool = False
    message: str = ""
    # Normalised (0..1) hand landmark list: [{"x": float, "y": float, "z": float}, ...]
    hand_landmarks: Optional[List[Dict[str, float]]] = field(default=None)


class GestureRecognizer:
    """Trained ISL gesture recognizer using MediaPipe hand crops + HOG + linear classifier.

    Compatible with MediaPipe >= 0.10 (Tasks API) which removed mp.solutions.
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        image_size: int = 64,
        confidence_threshold: float = 0.45,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Backend objects — created lazily on first predict()
        self._landmarker = None          # mediapipe.tasks HandLandmarker (new API ≥0.10)
        self._hog = self._build_hog_descriptor(image_size)
        self._classifier = None
        self._labels: List[str] = []
        self._trained_at = ""
        self._validation_accuracy = None
        self._sample_count = 0
        self._load_error = ""
        self._backend_error = ""
        self._load_artifacts()

    # ── public ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def stats(self) -> Dict[str, object]:
        return {
            "trained": self._classifier is not None,
            "label_count": len(self._labels),
            "labels": list(self._labels),
            "model_path": str(self.model_path),
            "labels_path": str(self.labels_path),
            "validation_accuracy": self._validation_accuracy,
            "sample_count": self._sample_count,
            "trained_at": self._trained_at,
            "load_error": self._load_error,
        }

    def predict(self, frame_bgr: np.ndarray) -> GesturePrediction:
        backend_ok, backend_msg = self._ensure_backend()
        if not backend_ok:
            return GesturePrediction(message=backend_msg)

        crop, has_hand, landmarks = self._extract_hand_crop_with_landmarks(frame_bgr)

        if self._classifier is None:
            message = self._load_error or "Trained ISL model not found. Please run train_isl_model.py."
            return GesturePrediction(message=message, has_hand=has_hand, hand_landmarks=landmarks)

        variants: List[np.ndarray] = [frame_bgr]
        if crop is not None:
            variants.insert(0, crop)

        best_candidate = None
        best_confidence = -1.0
        for variant in variants:
            features = self._extract_features(variant)
            proba = self._predict_proba(features)
            if proba is None or not len(proba):
                continue
            index = int(np.argmax(proba))
            confidence = float(proba[index])
            label = self._labels[index] if index < len(self._labels) else str(index)
            if confidence > best_confidence:
                best_confidence = confidence
                best_candidate = label

        if best_candidate is None:
            return GesturePrediction(
                message="Unable to score the gesture.",
                has_hand=has_hand,
                hand_landmarks=landmarks,
            )

        accepted = best_confidence >= self.confidence_threshold
        return GesturePrediction(
            candidate=best_candidate if accepted else "",
            accepted=accepted,
            confidence=best_confidence if accepted else 0.0,
            distance=None,
            has_hand=has_hand,
            message=f"Detected {best_candidate}." if accepted else "Low confidence gesture.",
            hand_landmarks=landmarks,
        )

    # ── private: artifact loading ─────────────────────────────────────────────

    def _load_artifacts(self) -> None:
        if not self.model_path.exists():
            self._labels = self._load_labels_fallback()
            return
        try:
            payload = joblib.load(self.model_path)
        except Exception as exc:
            self._load_error = (
                f"Unable to load ISL model from {self.model_path.name}: {exc}. "
                "The artifact likely needs to be retrained with the current scikit-learn version."
            )
            self._labels = self._load_labels_fallback()
            return

        self._classifier = payload.get("classifier")
        self._labels = list(payload.get("labels") or self._load_labels_fallback())
        self._trained_at = str(payload.get("trained_at") or "")
        self._validation_accuracy = payload.get("validation_accuracy")
        self._sample_count = int(payload.get("sample_count") or 0)

        if not self._labels and self.labels_path.exists():
            self._labels = self._load_labels_fallback()

    def _load_labels_fallback(self) -> List[str]:
        if not self.labels_path.exists():
            return []
        try:
            with self.labels_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return []
        if isinstance(raw, dict):
            labels = raw.get("labels")
            if isinstance(labels, list):
                return [str(item) for item in labels if str(item).strip()]
        if isinstance(raw, list):
            return [str(item) for item in raw if str(item).strip()]
        return []

    # ── private: MediaPipe backend (Tasks API ≥ 0.10) ────────────────────────

    def _ensure_backend(self) -> Tuple[bool, str]:
        if self._landmarker is not None:
            return True, ""
        if self._backend_error:
            return False, self._backend_error

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:
            self._backend_error = f"MediaPipe is not installed or incompatible: {exc}"
            return False, self._backend_error

        # Ensure the .task bundle is available locally
        task_path = _DEFAULT_TASK_PATH
        if not task_path.exists():
            try:
                task_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[GestureRecognizer] Downloading hand landmarker model to {task_path} …")
                urllib.request.urlretrieve(_HAND_LANDMARKER_URL, str(task_path))
                print("[GestureRecognizer] Download complete.")
            except Exception as exc:
                self._backend_error = (
                    f"Could not download hand landmarker model: {exc}. "
                    "Please manually place 'hand_landmarker.task' in the runtime/ folder."
                )
                return False, self._backend_error

        try:
            base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            return True, ""
        except Exception as exc:
            self._backend_error = f"Failed to initialise HandLandmarker: {exc}"
            return False, self._backend_error

    # ── private: landmark extraction ─────────────────────────────────────────

    def _extract_hand_crop_with_landmarks(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool, Optional[List[Dict[str, float]]]]:
        """Return (cropped_hand_image, has_hand, normalised_landmark_list)."""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision import HandLandmarker
        except Exception:
            return None, False, None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception:
            return None, False, None

        if not result.hand_landmarks:
            return None, False, None

        height, width = frame_bgr.shape[:2]
        raw_landmarks = result.hand_landmarks[0]

        # Normalised landmarks (0..1) for the browser canvas overlay
        lm_list: List[Dict[str, float]] = [
            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
            for lm in raw_landmarks
        ]

        xs = [lm.x * width  for lm in raw_landmarks]
        ys = [lm.y * height for lm in raw_landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        span   = max(x2 - x1, y2 - y1)
        margin = max(18.0, span * 0.35)
        left   = max(0,      int(x1 - margin))
        top    = max(0,      int(y1 - margin))
        right  = min(width,  int(x2 + margin))
        bottom = min(height, int(y2 + margin))

        if right <= left or bottom <= top:
            return None, True, lm_list

        crop = frame_bgr[top:bottom, left:right]
        if crop.size == 0:
            return None, True, lm_list
        return crop, True, lm_list

    # ── private: HOG feature extraction ──────────────────────────────────────

    def _build_hog_descriptor(self, image_size: int):
        return cv2.HOGDescriptor(
            _winSize=(image_size, image_size),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

    def _extract_features(self, image_bgr: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        features = self._hog.compute(resized)
        if features is None:
            return np.zeros((0,), dtype=np.float32)
        return features.reshape(-1).astype(np.float32)

    def _predict_proba(self, features: np.ndarray) -> Optional[np.ndarray]:
        if self._classifier is None:
            return None

        sample = features.reshape(1, -1)
        if hasattr(self._classifier, "predict_proba"):
            try:
                return np.asarray(self._classifier.predict_proba(sample)[0], dtype=np.float32)
            except Exception:
                pass

        if hasattr(self._classifier, "decision_function"):
            scores = np.asarray(self._classifier.decision_function(sample), dtype=np.float32)
            scores = scores.reshape(-1)
            exp    = np.exp(scores - np.max(scores))
            total  = float(np.sum(exp))
            if total > 0:
                return exp / total
        return None
