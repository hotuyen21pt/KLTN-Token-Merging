"""Ollama LLM segmenter for UOS."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Sequence

from uos.parsing import parse_units, validate_units
from uos.prompt import UOS_LINGUISTIC_PROMPT

MODEL_ALIASES = {
    "Qwen3-8B-Instruct": "qwen3:8b",
}
PROMPT_VERSION = "linguistic_v11"

# Allow overriding the Ollama endpoint (e.g. "http://ollama:11434" inside Docker).
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.strip(), name.strip())


def build_prompt(sentence: str) -> str:
    return UOS_LINGUISTIC_PROMPT.replace("{sentence}", sentence.strip())


class OllamaUOSSegmenter:
    def __init__(
        self,
        model_name: str = "Qwen3-8B-Instruct",
        host: str = DEFAULT_OLLAMA_HOST,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        timeout_s: int = 180,
    ) -> None:
        self.model_name = resolve_model_name(model_name)
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s

    def _call(self, prompt: str) -> str:
        payload: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_new_tokens},
        }
        if self.model_name.lower().startswith("qwen3"):
            payload["think"] = False
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if "Connection refused" in str(reason) or "61" in str(reason):
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.host}. "
                    "Start: open -a Ollama  OR  ollama serve"
                ) from e
            raise RuntimeError(
                f"Ollama failed ({self.host}, model={self.model_name}): {reason}"
            ) from e
        return str((data.get("message") or {}).get("content", ""))

    def segment_with_raw(self, sentence: str) -> tuple[List[str], str]:
        raw = self._call(build_prompt(sentence))
        units = parse_units(raw) or [sentence.strip()]
        return validate_units(sentence, units), raw

    def segment(self, sentence: str) -> List[str]:
        units, _ = self.segment_with_raw(sentence)
        return units

    def segment_batch(self, sentences: Sequence[str]) -> List[List[str]]:
        return [self.segment(s) for s in sentences]


def check_ollama(host: str = DEFAULT_OLLAMA_HOST, model_name: str = "Qwen3-8B-Instruct") -> None:
    resolved = resolve_model_name(model_name)
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {host}. Start: ollama serve"
        ) from e
    names = {m.get("name", "") for m in data.get("models", [])}
    if resolved not in names and f"{resolved}:latest" not in names:
        raise RuntimeError(f"Model {resolved!r} not found. Run: ollama pull {resolved}")


def create_llm_segmenter(model_name: str, **kwargs) -> OllamaUOSSegmenter:
    return OllamaUOSSegmenter(model_name=model_name, **kwargs)
