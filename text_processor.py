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

# Simple grammar-reduction list to make output closer to signer-friendly keyword order.
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
    """Cleans ASR text and converts it to a simplified ISL-like token stream."""

    def clean_text(self, text: str) -> str:
        lowered = text.lower().strip()

        # Remove filler words while preserving sentence order for meaningful terms.
        for filler in sorted(FILLER_WORDS, key=len, reverse=True):
            lowered = re.sub(rf"\b{re.escape(filler)}\b", " ", lowered)

        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def simplify_for_isl(self, text: str) -> str:
        words = text.split()
        simplified = [w for w in words if w not in AUXILIARY_WORDS]
        return " ".join(simplified)

    def tokenize(self, text: str) -> List[str]:
        return [tok for tok in text.split() if tok]
