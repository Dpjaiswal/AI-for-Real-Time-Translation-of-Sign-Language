import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .gesture_recognizer import GesturePrediction


@dataclass
class TranslationEntry:
    mode: str
    input_text: str
    output_text: str
    timestamp: str
    confidence: Optional[float] = None
    audio_url: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SessionState:
    session_id: str
    transcript: str = ""
    tokens: List[str] = field(default_factory=list)
    history: List[TranslationEntry] = field(default_factory=list)
    last_candidate: str = ""
    candidate_count: int = 0
    last_commit_time: float = 0.0

    def register_prediction(
        self,
        prediction: GesturePrediction,
        stable_frames: int,
        commit_interval_seconds: float,
        now_seconds: float,
    ) -> str:
        if not prediction.accepted or not prediction.candidate:
            self.last_candidate = ""
            self.candidate_count = 0
            return ""

        threshold = 1 if prediction.candidate.isdigit() or (len(prediction.candidate) == 1 and prediction.candidate.isalpha()) else max(2, stable_frames)

        if prediction.candidate == self.last_candidate:
            self.candidate_count += 1
        else:
            self.last_candidate = prediction.candidate
            self.candidate_count = 1

        if self.candidate_count < threshold:
            return ""

        if prediction.candidate == self.last_candidate and (now_seconds - self.last_commit_time) < commit_interval_seconds:
            return ""

        self.last_commit_time = now_seconds
        self.last_candidate = ""
        self.candidate_count = 0
        committed = prediction.candidate
        self.tokens.append(committed)
        self.transcript = self._render_transcript()
        return committed

    def _render_transcript(self) -> str:
        parts: List[str] = []
        for token in self.tokens:
            token = token.strip()
            if not token:
                continue
            if token in {"space", "<space>"}:
                if parts and not parts[-1].endswith(" "):
                    parts.append(" ")
                continue
            if token in {"backspace", "delete", "del"}:
                if parts:
                    parts.pop()
                continue
            if len(token) == 1 and token.isalnum():
                parts.append(token)
                continue
            if parts and not parts[-1].endswith(" "):
                parts.append(" ")
            parts.append(token)
        return "".join(parts).strip()

    def clear(self) -> None:
        self.transcript = ""
        self.tokens.clear()
        self.history.clear()
        self.last_candidate = ""
        self.candidate_count = 0
        self.last_commit_time = 0.0

    def add_history(self, entry: TranslationEntry) -> None:
        self.history.append(entry)


class SessionStore:
    def __init__(self, history_dir: str):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def clear(self, session_id: str) -> SessionState:
        session = self.get(session_id)
        session.clear()
        self._save(session)
        return session

    def record_entry(self, session_id: str, entry: TranslationEntry) -> SessionState:
        session = self.get(session_id)
        session.add_history(entry)
        self._save(session)
        return session

    def snapshot(self, session_id: str) -> Dict[str, object]:
        session = self.get(session_id)
        transcript = session.transcript or (session.history[-1].output_text if session.history else "")
        return {
            "session_id": session.session_id,
            "transcript": transcript,
            "tokens": list(session.tokens),
            "history": [asdict(item) for item in session.history],
            "token_count": len(session.tokens),
        }

    def _save(self, session: SessionState) -> None:
        output_path = self.history_dir / f"{session.session_id}.json"
        payload = {
            "session_id": session.session_id,
            "transcript": session.transcript or (session.history[-1].output_text if session.history else ""),
            "tokens": session.tokens,
            "history": [asdict(item) for item in session.history],
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
