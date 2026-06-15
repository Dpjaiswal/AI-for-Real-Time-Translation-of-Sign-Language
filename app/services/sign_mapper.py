import os
import re
from typing import Dict, List, Tuple


class SignMapper:
    """Map text tokens or phrases to reference sign videos when a matching clip exists."""

    def __init__(self, sign_dir: str = "sign_videos"):
        self.sign_dir = sign_dir
        self._video_index: Dict[str, str] = {}
        self.refresh_index()

    def refresh_index(self) -> None:
        self._video_index = self._build_video_index()

    def available_count(self) -> int:
        return len(self._video_index)

    def map_tokens(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        self.refresh_index()

        video_sequence: List[str] = []
        missing_words: List[str] = []
        i = 0

        while i < len(tokens):
            match_path, span = self._find_best_phrase(tokens, i)
            if match_path:
                video_sequence.append(match_path)
                i += span
                continue

            token = tokens[i]
            word_video = self._find_word_video(token)
            if word_video:
                video_sequence.append(word_video)
            else:
                missing_words.append(token)
                video_sequence.extend(self._spell_word_videos(token))
            i += 1

        return video_sequence, missing_words

    def map_text(self, text: str) -> Tuple[List[str], List[str]]:
        tokens = self._tokenize(text)
        return self.map_tokens(tokens)

    def first_video_for_text(self, text: str) -> str:
        video_sequence, _ = self.map_text(text)
        return video_sequence[0] if video_sequence else ""

    def _find_best_phrase(self, tokens: List[str], start_index: int) -> Tuple[str, int]:
        best_path = ""
        best_span = 0
        max_span = min(5, len(tokens) - start_index)

        for span in range(max_span, 1, -1):
            phrase = " ".join(tokens[start_index : start_index + span])
            selected = self._video_index.get(self._normalize_key(phrase), "")
            if selected:
                return selected, span

        return best_path, best_span

    def _find_word_video(self, token: str) -> str:
        return self._video_index.get(self._normalize_key(token), "")

    def _spell_word_videos(self, token: str) -> List[str]:
        paths: List[str] = []
        for ch in token:
            if not ch.isalpha():
                continue
            selected = self._video_index.get(self._normalize_key(ch), "")
            if selected:
                paths.append(selected)
        return paths

    def _build_video_index(self) -> Dict[str, str]:
        if not os.path.isdir(self.sign_dir):
            return {}

        index: Dict[str, str] = {}
        for file_name in os.listdir(self.sign_dir):
            if not file_name.lower().endswith(".mp4"):
                continue
            full_path = os.path.join(self.sign_dir, file_name)
            if not os.path.isfile(full_path):
                continue
            key = self._normalize_key(os.path.splitext(file_name)[0])
            if key and key not in index:
                index[key] = full_path
        return index

    @staticmethod
    def _normalize_key(value: str) -> str:
        lowered = value.lower().strip()
        lowered = lowered.replace("-", "_").replace(" ", "_")
        lowered = re.sub(r"[^a-z0-9_]", "", lowered)
        lowered = re.sub(r"_+", "_", lowered).strip("_")
        return lowered

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return [token for token in cleaned.split() if token]
