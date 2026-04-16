"""Prompt library for MEDLEY-BENCH three-step benchmark."""
import hashlib


def content_hash(text: str) -> str:
    """SHA-256 hash of prompt template content for version tracking."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
