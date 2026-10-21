"""Ollama housekeeping over its HTTP API (no SDK): free VRAM before other GPU stages."""

from __future__ import annotations

import httpx


def loaded_models(base_url: str) -> list[str]:
    try:
        return [m["name"] for m in httpx.get(f"{base_url}/api/ps", timeout=3).json()["models"]]
    except (httpx.HTTPError, KeyError, ValueError):
        return []


def unload_all(base_url: str) -> list[str]:
    """Ask Ollama to evict every loaded model (keep_alive=0). Returns the names unloaded."""
    names = loaded_models(base_url)
    for name in names:
        for endpoint, body in (
            ("/api/generate", {"model": name, "keep_alive": 0}),
            ("/api/embed", {"model": name, "input": "", "keep_alive": 0}),
        ):
            try:
                if httpx.post(f"{base_url}{endpoint}", json=body, timeout=30).is_success:
                    break
            except httpx.HTTPError:
                continue
    return names
