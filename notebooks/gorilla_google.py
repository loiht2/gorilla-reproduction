"""Google connection helpers extracted from the original notebook.

The key is read only when requested. Never display it or raw provider errors.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]

import re
import urllib.request
import urllib.error
import urllib.parse

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def read_google_key():
    # Supports a raw key, a provider-label/key pair, or a standard JSON/env field.
    try:
        raw = (root / "credentials").read_text(encoding="utf-8").strip()
        names = {"api_key", "google_api_key", "gemini_api_key", "key"}
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        candidates = []
        if isinstance(data, dict):
            candidates = [v for k, v in data.items()
                          if k.lower() in names and isinstance(v, str)]
        else:
            lines = [line.strip().strip("`\"'") for line in raw.splitlines() if line.strip()]
            if len(lines) == 2 and re.search(r"google|gemini|ai.?studio", lines[0], re.I):
                candidates = [lines[1]]
            elif len(lines) == 1:
                match = re.fullmatch(r"(?:export\s+)?([A-Za-z_]+)\s*[:=]\s*(.+)", lines[0])
                candidates = ([match[2].strip().strip("\"'")]
                              if match and match[1].lower() in names else lines)
        candidates = list(set(candidates))
        if len(candidates) == 1 and re.fullmatch(r"[A-Za-z0-9._~+=/-]{16,4096}", candidates[0]):
            return candidates[0]
    except Exception:
        pass
    raise RuntimeError("Unable to read one Google key; credential details suppressed.") from None

def google_request(endpoint, payload=None):
    if not re.fullmatch(r"models(?:/[A-Za-z0-9._-]+:generateContent)?(?:\?pageToken=[A-Za-z0-9%._~-]+)?", endpoint):
        return {"ok": False, "error": "INVALID_ENDPOINT"}
    try:
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/" + endpoint,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"x-goog-api-key": read_google_key(), "Content-Type": "application/json"},
        )
        with urllib.request.build_opener(NoRedirect).open(request, timeout=60) as response:
            return {"ok": True, "http_status": response.status, "data": json.load(response)}
    except urllib.error.HTTPError as error:
        return {"ok": False, "error": "HTTP_ERROR", "http_status": error.code}
    except Exception:
        return {"ok": False, "error": "CREDENTIAL_NETWORK_OR_RESPONSE_ERROR"}
