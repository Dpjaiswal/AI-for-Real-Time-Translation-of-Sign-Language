from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency during bootstrap
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in {None, ""} else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in {None, ""} else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in {None, ""} else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((BASE_DIR / path).resolve())


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    debug: bool
    templates_path: str
    isl_dataset_dir: str
    isl_model_path: str
    isl_labels_path: str
    isl_image_size: int
    isl_samples_path: str
    sign_dir: str
    runtime_dir: str
    language: str
    match_threshold: float
    stable_frames: int
    commit_interval_seconds: float
    tts_rate: int
    tts_backend: str
    tts_voice: str
    tts_edge_rate: str
    tts_edge_pitch: str
    sign_model_endpoint: str
    sign_model_timeout_seconds: float
    sign_capture_interval_ms: int
    recording_seconds: int
    camera_index: int
    camera_facing_mode: str
    whisper_model: str
    mediapipe_min_detection_confidence: float
    mediapipe_min_tracking_confidence: float

    @property
    def runtime_path(self) -> Path:
        return Path(self.runtime_dir)

    @property
    def audio_cache_dir(self) -> Path:
        return self.runtime_path / "audio"

    @property
    def history_dir(self) -> Path:
        return self.runtime_path / "history"


def load_config() -> AppConfig:
    return AppConfig(
        host=_env("APP_HOST", "127.0.0.1"),
        port=_env_int("APP_PORT", 5000),
        debug=_env_bool("APP_DEBUG", False),
        templates_path=_resolve_path(_env("TEMPLATES_PATH", "app/data/sign_templates.example.json")),
        isl_dataset_dir=_resolve_path(_env("ISL_DATASET_DIR", "Indian")),
        isl_model_path=_resolve_path(_env("ISL_MODEL_PATH", "runtime/isl_model.joblib")),
        isl_labels_path=_resolve_path(_env("ISL_LABELS_PATH", "runtime/isl_labels.json")),
        isl_image_size=_env_int("ISL_IMAGE_SIZE", 64),
        isl_samples_path=_resolve_path(_env("ISL_SAMPLES_PATH", "runtime/isl_samples.json")),
        sign_dir=_resolve_path(_env("SIGN_DIR", "sign_videos")),
        runtime_dir=_resolve_path(_env("RUNTIME_DIR", "runtime")),
        language=_env("APP_LANGUAGE", "en"),
        match_threshold=_env_float("MATCH_THRESHOLD", 0.16),
        stable_frames=_env_int("STABLE_FRAMES", 2),
        commit_interval_seconds=_env_float("COMMIT_INTERVAL_SECONDS", 0.35),
        tts_rate=_env_int("TTS_RATE", 175),
        tts_backend=_env("TTS_BACKEND", "edge"),
        tts_voice=_env("TTS_VOICE", "en-IN-NeerjaNeural"),
        tts_edge_rate=_env("TTS_EDGE_RATE", "+0%"),
        tts_edge_pitch=_env("TTS_EDGE_PITCH", "+0Hz"),
        sign_model_endpoint=_env("SIGN_MODEL_ENDPOINT", ""),
        sign_model_timeout_seconds=_env_float("SIGN_MODEL_TIMEOUT_SECONDS", 12.0),
        sign_capture_interval_ms=_env_int("SIGN_CAPTURE_INTERVAL_MS", 350),
        recording_seconds=_env_int("RECORDING_SECONDS", 5),
        camera_index=_env_int("CAMERA_INDEX", 0),
        camera_facing_mode=_env("CAMERA_FACING_MODE", "user"),
        whisper_model=_env("WHISPER_MODEL", "base"),
        mediapipe_min_detection_confidence=_env_float("MEDIAPIPE_MIN_DETECTION_CONFIDENCE", 0.6),
        mediapipe_min_tracking_confidence=_env_float("MEDIAPIPE_MIN_TRACKING_CONFIDENCE", 0.6),
    )
