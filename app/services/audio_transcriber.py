import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class AudioTranscriber:
    """Transcribe recorded speech using SpeechRecognition first, then Whisper fallback."""

    def __init__(self, language: str = "en", whisper_model: str = "base"):
        self.language = language
        self.whisper_model = whisper_model

    def transcribe_file(self, input_path: str) -> str:
        wav_path = self._prepare_wav(input_path)

        text = self._transcribe_with_speech_recognition(wav_path)
        if text:
            return text

        return self._transcribe_with_whisper(wav_path)

    def _prepare_wav(self, input_path: str) -> str:
        suffix = Path(input_path).suffix.lower()
        if suffix in {".wav", ".flac", ".aiff", ".aif"}:
            return input_path

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "Audio upload is not in a WAV/FLAC/AIFF format and ffmpeg was not found. "
                "Install ffmpeg or upload a WAV file."
            )

        temp_dir = tempfile.mkdtemp(prefix="signlang_audio_")
        wav_path = os.path.join(temp_dir, "converted.wav")
        command = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            wav_path,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    def _transcribe_with_speech_recognition(self, wav_path: str) -> str:
        try:
            import speech_recognition as sr
        except Exception:
            return ""

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
        except Exception:
            return ""

        for backend in (self._recognize_sphinx, self._recognize_google):
            try:
                text = backend(recognizer, audio)
                if text:
                    return text.strip()
            except Exception:
                continue

        return ""

    def _recognize_sphinx(self, recognizer, audio) -> str:
        try:
            return recognizer.recognize_sphinx(audio, language=self.language)
        except Exception:
            return ""

    def _recognize_google(self, recognizer, audio) -> str:
        try:
            return recognizer.recognize_google(audio, language=self.language)
        except Exception:
            return ""

    def _transcribe_with_whisper(self, wav_path: str) -> str:
        try:
            import whisper
        except Exception:
            return ""

        model = whisper.load_model(self.whisper_model)
        result = model.transcribe(
            wav_path,
            language=self.language,
            fp16=False,
            condition_on_previous_text=False,
            verbose=False,
        )
        return (result.get("text") or "").strip()
