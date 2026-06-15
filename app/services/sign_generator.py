class SignGenerator:
    """Generate sign-language playback data using a remote model or a local storyboard fallback."""

    def __init__(self, endpoint: str = "", sign_dir: str = "", timeout_seconds: float = 12.0):
        self.endpoint = (endpoint or "").strip()
        self.sign_dir = sign_dir
        self.timeout_seconds = timeout_seconds

        from .sign_mapper import SignMapper

        self.mapper = SignMapper(sign_dir) if sign_dir else None

    def generate(self, text: str, language: str = "en", session_id: str = "anonymous"):
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text is required for sign generation.")

        # 1. Try local video mapping first if sign_dir is provided
        if self.mapper and self.mapper.available_count() > 0:
            video_path = self.mapper.first_video_for_text(clean_text)
            if video_path:
                import os
                # We return a relative URL that the server will handle
                return self._normalize_response(
                    {
                        "video_url": f"/signs/{os.path.basename(video_path)}",
                        "gloss": clean_text,
                        "poster_text": clean_text,
                    },
                    fallback_text=clean_text,
                    backend="local",
                )

        # 2. Try remote model if endpoint is provided
        if self.endpoint:
            remote = self._generate_remote(clean_text, language=language, session_id=session_id)
            if remote is not None:
                return remote

        # 3. Fallback to storyboard
        return self._generate_storyboard(clean_text, language=language)

    def _generate_remote(self, text: str, language: str, session_id: str):
        import json
        from urllib import error, request

        payload = json.dumps(
            {
                "text": text,
                "language": language,
                "session_id": session_id,
            }
        ).encode("utf-8")

        req = request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except Exception:
            return None

        return self._normalize_response(data, fallback_text=text, backend="remote")

    def _generate_storyboard(self, text: str, language: str):
        tokens = self._tokenize(text)
        steps = []
        for token in tokens:
            steps.append(
                {
                    "label": token,
                    "duration_ms": 900,
                    "description": f"Visual prompt for {token}",
                }
            )

        return self._normalize_response(
            {
                "video_url": "",
                "steps": steps,
                "gloss": " ".join(tokens),
                "poster_text": text,
            },
            fallback_text=text,
            backend="storyboard",
        )

    def _normalize_response(self, data, fallback_text: str, backend: str):
        if not isinstance(data, dict):
            data = {}

        video_url = (data.get("video_url") or data.get("media_url") or "").strip()
        gloss = (data.get("gloss") or data.get("output_text") or fallback_text or "").strip()
        poster_text = (data.get("poster_text") or gloss or fallback_text or "").strip()
        steps = data.get("steps")
        if not isinstance(steps, list):
            steps = []

        normalized_steps = []
        for item in steps:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("text") or "").strip()
            if not label:
                continue
            normalized_steps.append(
                {
                    "label": label,
                    "duration_ms": int(item.get("duration_ms") or item.get("duration") or 900),
                    "description": str(item.get("description") or "").strip(),
                }
            )

        if not normalized_steps and poster_text:
            for token in self._tokenize(poster_text):
                normalized_steps.append(
                    {
                        "label": token,
                        "duration_ms": 900,
                        "description": f"Visual prompt for {token}",
                    }
                )

        return SignGenerationResult(
            backend=backend,
            video_url=video_url,
            poster_text=poster_text,
            gloss=gloss,
            steps=normalized_steps,
        )

    @staticmethod
    def _tokenize(text: str):
        import re

        cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return [token for token in cleaned.split() if token]


class SignGenerationResult:
    def __init__(self, backend: str, video_url: str, poster_text: str, gloss: str, steps):
        self.backend = backend
        self.video_url = video_url
        self.poster_text = poster_text
        self.gloss = gloss
        self.steps = steps
