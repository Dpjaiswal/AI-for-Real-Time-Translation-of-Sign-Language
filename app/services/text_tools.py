import re
from typing import List


FILLER_WORDS = {
    "um",
    "uh",
    "ah",
    "er",
    "hmm",
    "okay",
    "ok",
    "like",
    "you know",
    "actually",
    "basically",
}

AUXILIARY_WORDS = {
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "that",
    "this",
    "it",
}


class TextProcessor:
    """Normalize ASR text and make it slightly sign-language friendly."""

    def clean_text(self, text: str) -> str:
        lowered = text.lower().strip()
        for filler in sorted(FILLER_WORDS, key=len, reverse=True):
            lowered = re.sub(rf"\b{re.escape(filler)}\b", " ", lowered)
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def simplify_for_isl(self, text: str) -> str:
        words = text.split()
        simplified = [word for word in words if word not in AUXILIARY_WORDS]
        return " ".join(simplified)

    def tokenize(self, text: str) -> List[str]:
        return [token for token in text.split() if token]
