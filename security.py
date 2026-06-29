"""
Security utilities: rate limiting, API key auth, security headers, file validation.
"""
import os
import re
import logging
from typing import Optional
from pathlib import Path

from fastapi import HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter

logger = logging.getLogger(__name__)


def _get_client_ip(request: Request) -> str:
    """Return real client IP, honouring X-Forwarded-For from reverse proxies."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_get_client_ip)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """Enforce API key when API_KEY env var is set; no-op otherwise."""
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response


MAX_FILE_SIZE_BYTES = int(os.environ.get("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024

ALLOWED_EXTENSIONS = frozenset({
    ".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg",
    ".webm", ".avi", ".mov", ".mkv", ".aac", ".wma", ".opus",
})

ALLOWED_MODELS = frozenset({
    "tiny", "base", "small", "medium", "large", "large-v2", "large-v3",
})

_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")


def validate_upload(filename: str) -> str:
    """Validate filename/extension; return safe basename or raise 400."""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return safe_name


def validate_language(language: Optional[str]) -> None:
    if language is not None and not _LANGUAGE_RE.match(language):
        raise HTTPException(
            status_code=400,
            detail="Invalid language code. Use ISO 639-1/2 codes such as 'en', 'fr', 'deu'.",
        )


def validate_model(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Allowed: {', '.join(sorted(ALLOWED_MODELS))}",
        )
