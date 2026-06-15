import argparse

from app.config import AppConfig, load_config
from app.web.server import create_app


def parse_args() -> argparse.Namespace:
    base = load_config()
    parser = argparse.ArgumentParser(description="Real-time AI sign language translation system")
    parser.add_argument("--host", default=base.host, help="Host to bind the Flask server.")
    parser.add_argument("--port", type=int, default=base.port, help="Port to bind the Flask server.")
    parser.add_argument("--debug", action="store_true", default=base.debug, help="Run Flask in debug mode.")
    parser.add_argument("--templates", default=base.templates_path, help="Path to the gesture template JSON file.")
    parser.add_argument("--isl-dataset", default=base.isl_dataset_dir, help="Path to the ISL dataset root folder.")
    parser.add_argument("--isl-model", default=base.isl_model_path, help="Path to the trained ISL model file.")
    parser.add_argument("--isl-labels", default=base.isl_labels_path, help="Path to the trained ISL labels file.")
    parser.add_argument("--isl-image-size", type=int, default=base.isl_image_size, help="Image size used for ISL inference.")
    parser.add_argument("--isl-samples", default=base.isl_samples_path, help="Path to the optional ISL sample store.")
    parser.add_argument("--sign-dir", default=base.sign_dir, help="Directory with optional reference sign videos.")
    parser.add_argument("--runtime-dir", default=base.runtime_dir, help="Directory for cached audio and history files.")
    parser.add_argument("--language", default=base.language, help="Language code used by speech transcription.")
    parser.add_argument("--match-threshold", type=float, default=base.match_threshold, help="Gesture match threshold.")
    parser.add_argument("--stable-frames", type=int, default=base.stable_frames, help="Frames required before commit.")
    parser.add_argument("--commit-interval-seconds", type=float, default=base.commit_interval_seconds, help="Cooldown between commits.")
    parser.add_argument("--tts-rate", type=int, default=base.tts_rate, help="Text-to-speech rate.")
    parser.add_argument("--tts-backend", default=base.tts_backend, help="Text-to-speech backend.")
    parser.add_argument("--tts-voice", default=base.tts_voice, help="Edge TTS voice name.")
    parser.add_argument("--tts-edge-rate", default=base.tts_edge_rate, help="Edge TTS rate string.")
    parser.add_argument("--tts-edge-pitch", default=base.tts_edge_pitch, help="Edge TTS pitch string.")
    parser.add_argument("--sign-model-endpoint", default=base.sign_model_endpoint, help="Remote sign-generation model endpoint.")
    parser.add_argument("--sign-model-timeout-seconds", type=float, default=base.sign_model_timeout_seconds, help="Remote sign model timeout.")
    parser.add_argument("--sign-capture-interval-ms", type=int, default=base.sign_capture_interval_ms, help="Delay between webcam captures in milliseconds.")
    parser.add_argument("--recording-seconds", type=int, default=base.recording_seconds, help="Microphone recording duration.")
    parser.add_argument("--camera-index", type=int, default=base.camera_index, help="Browser-side camera index hint.")
    parser.add_argument("--camera-facing-mode", default=base.camera_facing_mode, help="Browser camera facing mode hint.")
    parser.add_argument("--whisper-model", default=base.whisper_model, help="Whisper model name for speech fallback.")
    parser.add_argument(
        "--mediapipe-min-detection-confidence",
        type=float,
        default=base.mediapipe_min_detection_confidence,
        help="Minimum MediaPipe detection confidence.",
    )
    parser.add_argument(
        "--mediapipe-min-tracking-confidence",
        type=float,
        default=base.mediapipe_min_tracking_confidence,
        help="Minimum MediaPipe tracking confidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        templates_path=args.templates,
        isl_dataset_dir=args.isl_dataset,
        isl_model_path=args.isl_model,
        isl_labels_path=args.isl_labels,
        isl_image_size=args.isl_image_size,
        isl_samples_path=args.isl_samples,
        sign_dir=args.sign_dir,
        runtime_dir=args.runtime_dir,
        language=args.language,
        match_threshold=args.match_threshold,
        stable_frames=args.stable_frames,
        commit_interval_seconds=args.commit_interval_seconds,
        tts_rate=args.tts_rate,
        tts_backend=args.tts_backend,
        tts_voice=args.tts_voice,
        tts_edge_rate=args.tts_edge_rate,
        tts_edge_pitch=args.tts_edge_pitch,
        sign_model_endpoint=args.sign_model_endpoint,
        sign_model_timeout_seconds=args.sign_model_timeout_seconds,
        sign_capture_interval_ms=args.sign_capture_interval_ms,
        recording_seconds=args.recording_seconds,
        camera_index=args.camera_index,
        camera_facing_mode=args.camera_facing_mode,
        whisper_model=args.whisper_model,
        mediapipe_min_detection_confidence=args.mediapipe_min_detection_confidence,
        mediapipe_min_tracking_confidence=args.mediapipe_min_tracking_confidence,
    )
    app = create_app(config)
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)


if __name__ == "__main__":
    main()
