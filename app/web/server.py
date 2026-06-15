import atexit
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from app.config import AppConfig
from app.services.audio_transcriber import AudioTranscriber
from app.services.gesture_recognizer import GestureRecognizer
from app.services.session_store import SessionStore, TranslationEntry
from app.services.sign_generator import SignGenerator
from app.services.text_tools import TextProcessor
from app.services.tts_service import TextToSpeechService


def _frame_from_upload(file_obj) -> np.ndarray | None:
    raw = np.frombuffer(file_obj.read(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def create_app(config: AppConfig) -> Flask:
    config.runtime_path.mkdir(parents=True, exist_ok=True)
    config.audio_cache_dir.mkdir(parents=True, exist_ok=True)
    config.history_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )

    gesture_recognizer = GestureRecognizer(
        config.isl_model_path,
        config.isl_labels_path,
        image_size=config.isl_image_size,
        confidence_threshold=config.match_threshold,
        min_detection_confidence=config.mediapipe_min_detection_confidence,
        min_tracking_confidence=config.mediapipe_min_tracking_confidence,
    )
    text_processor = TextProcessor()
    tts_service = TextToSpeechService(
        str(config.audio_cache_dir),
        rate=config.tts_rate,
        backend=config.tts_backend,
        voice=config.tts_voice,
        edge_rate=config.tts_edge_rate,
        edge_pitch=config.tts_edge_pitch,
    )
    sign_generator = SignGenerator(
        endpoint=config.sign_model_endpoint,
        sign_dir=config.sign_dir,
        timeout_seconds=config.sign_model_timeout_seconds,
    )
    sessions = SessionStore(str(config.history_dir))

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            app_config={
                "host": config.host,
                "port": config.port,
                "debug": config.debug,
                "templates_path": config.templates_path,
                "isl_dataset_dir": config.isl_dataset_dir,
                "isl_model_path": config.isl_model_path,
                "isl_labels_path": config.isl_labels_path,
                "isl_image_size": config.isl_image_size,
                "isl_samples_path": config.isl_samples_path,
                "sign_dir": config.sign_dir,
                "runtime_dir": config.runtime_dir,
                "language": config.language,
                "match_threshold": config.match_threshold,
                "stable_frames": config.stable_frames,
                "commit_interval_seconds": config.commit_interval_seconds,
                "tts_rate": config.tts_rate,
                "tts_backend": config.tts_backend,
                "tts_voice": config.tts_voice,
                "tts_edge_rate": config.tts_edge_rate,
                "tts_edge_pitch": config.tts_edge_pitch,
                "sign_model_endpoint": config.sign_model_endpoint,
                "sign_model_timeout_seconds": config.sign_model_timeout_seconds,
                "sign_capture_interval_ms": config.sign_capture_interval_ms,
                "recording_seconds": config.recording_seconds,
                "camera_index": config.camera_index,
                "camera_facing_mode": config.camera_facing_mode,
                "whisper_model": config.whisper_model,
                "mediapipe_min_detection_confidence": config.mediapipe_min_detection_confidence,
                "mediapipe_min_tracking_confidence": config.mediapipe_min_tracking_confidence,
                "gesture_stats": gesture_recognizer.stats(),
            },
        )

    @app.route("/media/<path:filename>")
    def media_file(filename: str):
        return send_from_directory(config.audio_cache_dir, filename, as_attachment=False)

    @app.route("/signs/<path:filename>")
    def sign_file(filename: str):
        return send_from_directory(config.sign_dir, filename, as_attachment=False)

    @app.route("/api/history")
    def api_history():
        session_id = (request.args.get("session_id") or "anonymous").strip() or "anonymous"
        return jsonify(sessions.snapshot(session_id))

    @app.route("/api/history/clear", methods=["POST"])
    def api_history_clear():
        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or request.form.get("session_id") or "anonymous").strip() or "anonymous"
        sessions.clear(session_id)
        return jsonify({"ok": True, "session_id": session_id})

    @app.route("/api/sign/recognize", methods=["POST"])
    def api_sign_recognize():
        session_id = (request.form.get("session_id") or request.headers.get("X-Session-Id") or "anonymous").strip() or "anonymous"
        file_obj = request.files.get("frame")
        if file_obj is None:
            return jsonify({"error": "Missing frame upload."}), 400

        frame = _frame_from_upload(file_obj)
        if frame is None:
            return jsonify({"error": "Unable to decode the frame."}), 400

        prediction = gesture_recognizer.predict(frame)
        session = sessions.get(session_id)
        committed = session.register_prediction(
            prediction,
            stable_frames=config.stable_frames,
            commit_interval_seconds=config.commit_interval_seconds,
            now_seconds=time.time(),
        )

        audio_url = ""
        if committed:
            tts_path = tts_service.synthesize(committed)
            audio_url = url_for("media_file", filename=tts_path.name)

            sessions.record_entry(
                session_id,
                TranslationEntry(
                    mode="sign",
                    input_text=prediction.candidate,
                    output_text=session.transcript,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    confidence=prediction.confidence,
                    audio_url=audio_url,
                    metadata={},
                ),
            )

        return jsonify(
            {
                "session_id": session_id,
                "candidate": prediction.candidate,
                "accepted": prediction.accepted,
                "confidence": prediction.confidence,
                "distance": prediction.distance,
                "has_hand": prediction.has_hand,
                "message": prediction.message,
                "committed": committed,
                "transcript": session.transcript,
                "audio_url": audio_url,
                "history": sessions.snapshot(session_id)["history"],
                "model_stats": gesture_recognizer.stats(),
                # Normalised (0..1) hand landmark list for browser canvas overlay
                "hand_landmarks": prediction.hand_landmarks or [],
            }
        )

    @app.route("/api/audio/transcribe", methods=["POST"])
    def api_audio_transcribe():
        session_id = (request.form.get("session_id") or request.headers.get("X-Session-Id") or "anonymous").strip() or "anonymous"
        language = (request.form.get("language") or config.language).strip() or config.language
        file_obj = request.files.get("audio")
        if file_obj is None:
            return jsonify({"error": "Missing audio upload."}), 400

        temp_dir = Path(tempfile.mkdtemp(prefix="signlang_upload_"))
        input_path = temp_dir / (file_obj.filename or "audio.webm")
        file_obj.save(input_path)

        try:
            transcriber = AudioTranscriber(language=language, whisper_model=config.whisper_model)
            text = transcriber.transcribe_file(str(input_path))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        if not text:
            return jsonify({"error": "No speech could be transcribed."}), 422

        cleaned = text_processor.clean_text(text)
        simplified = text_processor.simplify_for_isl(cleaned)
        session = sessions.get(session_id)
        output_text = simplified or cleaned or text
        session.transcript = output_text
        audio_url = ""

        if output_text:
            try:
                tts_path = tts_service.synthesize(output_text)
                audio_url = url_for("media_file", filename=tts_path.name)
            except Exception:
                audio_url = ""

        sessions.record_entry(
            session_id,
            TranslationEntry(
                mode="audio",
                input_text=text,
                output_text=output_text,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                audio_url=audio_url,
                metadata={"cleaned": cleaned, "simplified": simplified},
            ),
        )

        return jsonify(
            {
                "session_id": session_id,
                "raw_text": text,
                "cleaned_text": cleaned,
                "output_text": output_text,
                "audio_url": audio_url,
                "history": sessions.snapshot(session_id)["history"],
            }
        )

    @app.route("/api/text/speak", methods=["POST"])
    def api_text_speak():
        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or request.form.get("session_id") or "anonymous").strip() or "anonymous"
        text = (payload.get("text") or request.form.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Text is required."}), 400

        cleaned = text_processor.clean_text(text)
        simplified = text_processor.simplify_for_isl(cleaned)
        output_text = simplified or cleaned or text
        session = sessions.get(session_id)
        session.transcript = output_text

        try:
            tts_path = tts_service.synthesize(output_text)
            audio_url = url_for("media_file", filename=tts_path.name)
        except Exception as exc:
            return jsonify({"error": f"TTS failed: {exc}"}), 400

        sessions.record_entry(
            session_id,
            TranslationEntry(
                mode="text",
                input_text=text,
                output_text=output_text,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                audio_url=audio_url,
                metadata={"cleaned": cleaned, "simplified": simplified},
            ),
        )

        return jsonify(
            {
                "session_id": session_id,
                "input_text": text,
                "cleaned_text": cleaned,
                "output_text": output_text,
                "audio_url": audio_url,
                "history": sessions.snapshot(session_id)["history"],
            }
        )

    @app.route("/api/text/sign", methods=["POST"])
    def api_text_sign():
        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or request.form.get("session_id") or "anonymous").strip() or "anonymous"
        text = (payload.get("text") or request.form.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Text is required."}), 400

        cleaned = text_processor.clean_text(text)
        simplified = text_processor.simplify_for_isl(cleaned)
        output_text = simplified or cleaned or text
        generation = sign_generator.generate(output_text, language=config.language, session_id=session_id)

        sessions.record_entry(
            session_id,
            TranslationEntry(
                mode="text_sign",
                input_text=text,
                output_text=output_text,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metadata={
                    "cleaned": cleaned,
                    "simplified": simplified,
                    "backend": generation.backend,
                    "video_url": generation.video_url,
                    "gloss": generation.gloss,
                },
            ),
        )

        return jsonify(
            {
                "session_id": session_id,
                "input_text": text,
                "cleaned_text": cleaned,
                "output_text": output_text,
                "backend": generation.backend,
                "video_url": generation.video_url,
                "poster_text": generation.poster_text,
                "steps": generation.steps,
                "gloss": generation.gloss,
                "history": sessions.snapshot(session_id)["history"],
            }
        )

    @app.route("/api/health")
    def api_health():
        return jsonify(
            {
                "ok": True,
                "templates": config.templates_path,
                "model_endpoint": config.sign_model_endpoint,
                "isl_model": config.isl_model_path,
                "gesture_stats": gesture_recognizer.stats(),
            }
        )

    atexit.register(gesture_recognizer.close)
    return app
