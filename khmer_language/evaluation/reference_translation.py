"""Reference translation, for sanity-checking Khmer against an external source.

**Read this before trusting the output.** Machine translation is a weak
oracle for evaluating a language model, because MT systems are built to
always return something plausible. Feed one a nonsense Khmer string and
it will not say "this is nonsense" - it will produce fluent-looking
English, which is easy to mistake for evidence that the Khmer was fine.
So a readable translation does NOT prove the Khmer is correct.

What it IS reliable for, and what this module is used for here:

  - checking that Khmer text *written deliberately* says what it was
    meant to say (verifying authored benchmark cases, for instance)
  - spotting output that is obviously not Khmer at all
  - giving a non-Khmer reader a rough gloss of what a model produced

For judging whether generated Khmer is natural or correct, a native
speaker remains the authority - which is exactly why
`error_analyzer.py` reports naturalness as UNAVAILABLE rather than
inventing a score.

Uses MyMemory's free, documented endpoint. No API key, but it is rate
limited, so results are cached on disk and requests are spaced out.
Every failure mode returns None rather than raising: a translation
service being unreachable must never break evaluation.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ENDPOINT = "https://api.mymemory.translated.net/get"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "translations.json"
MIN_REQUEST_INTERVAL = 1.0  # seconds between calls, to stay a good citizen


@dataclass(frozen=True)
class Translation:
    source: str
    translated: str
    match: float
    from_cache: bool

    @property
    def low_confidence(self) -> bool:
        """MyMemory reports a match score; low values usually mean it fell
        back to a poor or partial match rather than a real translation."""
        return self.match < 0.5


class ReferenceTranslator:
    def __init__(
        self,
        cache_path: str | Path | None = DEFAULT_CACHE,
        timeout: float = 20.0,
        offline: bool = False,
    ):
        self.cache_path = Path(cache_path) if cache_path else None
        self.timeout = timeout
        self.offline = offline
        self._cache: dict[str, dict] = self._load_cache()
        self._last_request = 0.0

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path and self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _fetch(self, text: str, langpair: str) -> dict | None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        query = urllib.parse.urlencode({"q": text, "langpair": langpair})
        try:
            with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        finally:
            self._last_request = time.monotonic()

        data = payload.get("responseData") or {}
        translated = data.get("translatedText")
        if not isinstance(translated, str) or not translated:
            return None
        try:
            match = float(data.get("match", 0.0))
        except (TypeError, ValueError):
            match = 0.0
        return {"translated": translated, "match": match}

    def translate(self, text: str, source: str = "km", target: str = "en") -> Translation | None:
        """Translate `text`, or return None if unavailable.

        Returns None rather than raising, so evaluation still runs with no
        network. Callers must handle None instead of assuming a result.
        """
        text = text.strip()
        if not text:
            return None

        langpair = f"{source}|{target}"
        key = f"{langpair}:{text}"

        if key in self._cache:
            entry = self._cache[key]
            return Translation(text, entry["translated"], entry["match"], from_cache=True)

        if self.offline:
            return None

        entry = self._fetch(text, langpair)
        if entry is None:
            return None

        self._cache[key] = entry
        self._save_cache()
        return Translation(text, entry["translated"], entry["match"], from_cache=False)

    def round_trip(self, text: str, via: str = "en") -> Translation | None:
        """Translate Khmer -> `via` -> Khmer.

        Round-tripping is sometimes suggested as an automatic quality
        check, on the theory that meaningful text survives it. Treat the
        result as a hint at best: MT is lossy in both directions, so
        perfectly good Khmer routinely comes back altered, and nonsense
        can come back looking tidier than it went in.
        """
        forward = self.translate(text, "km", via)
        if forward is None:
            return None
        return self.translate(forward.translated, via, "km")
