import asyncio
import hashlib
import threading
from pathlib import Path
from typing import Optional


class TextToSpeechService:
    """Render text to cached audio files using Edge TTS first, then pyttsx3 fallback."""

    def __init__(
        self,
        output_dir: str,
        rate: int = 175,
        backend: str = "edge",
        voice: str = "en-IN-NeerjaNeural",
        edge_rate: str = "+0%",
        edge_pitch: str = "+0Hz",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.backend = (backend or "edge").strip().lower()
        self.voice = voice.strip()
        self.edge_rate = edge_rate.strip()
        self.edge_pitch = edge_pitch.strip()
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> Path:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text-to-speech requires non-empty text.")

        suffix = ".mp3" if self._should_use_edge() else ".wav"
        digest = hashlib.sha1(
            f"{self.backend}|{self.voice}|{self.edge_rate}|{self.edge_pitch}|{clean_text}".encode("utf-8")
        ).hexdigest()[:16]
        output_path = self.output_dir / f"tts_{digest}{suffix}"
        if output_path.exists():
            return output_path

        if self._should_use_edge():
            try:
                self._synthesize_with_edge(clean_text, output_path)
                return output_path
            except Exception:
                if self.backend == "edge":
                    raise

        return self._synthesize_with_pyttsx3(clean_text, output_path.with_suffix(".wav"))

    def _should_use_edge(self) -> bool:
        return self.backend in {"edge", "auto"}

    def _synthesize_with_edge(self, text: str, output_path: Path) -> None:
        try:
            import edge_tts
        except Exception as exc:
            raise RuntimeError(f"edge-tts is not available: {exc}") from exc

        async def _run() -> None:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.edge_rate,
                pitch=self.edge_pitch,
            )
            await communicate.save(str(output_path))

        asyncio.run(_run())

    def _synthesize_with_pyttsx3(self, text: str, output_path: Path) -> Path:
        try:
            import pyttsx3
        except Exception as exc:
            raise RuntimeError(f"pyttsx3 is not available: {exc}") from exc

        with self._lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()

        return output_path
