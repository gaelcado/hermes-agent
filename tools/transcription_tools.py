#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription using OpenAI's Whisper API.
Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, and Slack.

Supported models:
  - whisper-1        (cheapest, good quality)
  - gpt-4o-mini-transcribe  (better quality, higher cost)
  - gpt-4o-transcribe       (best quality, highest cost)

Supported input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg

Usage:
    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio("/path/to/audio.ogg")
    if result["success"]:
        print(result["transcript"])
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from tools.managed_tool_gateway import resolve_managed_tool_gateway

logger = logging.getLogger(__name__)


# Default STT model -- cheapest and widely available
DEFAULT_STT_MODEL = "whisper-1"

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}

# Maximum file size (25MB - OpenAI limit)
MAX_FILE_SIZE = 25 * 1024 * 1024


def transcribe_audio(file_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio file using OpenAI's Whisper API.

    This function calls the OpenAI Audio Transcriptions endpoint directly
    (not via OpenRouter, since Whisper isn't available there).

    Args:
        file_path: Absolute path to the audio file to transcribe.
        model:     Whisper model to use. Defaults to config or "whisper-1".

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
    """
    try:
        api_key, base_url = _resolve_openai_audio_client_config()
    except ValueError as exc:
        return {
            "success": False,
            "transcript": "",
            "error": str(exc),
        }

    audio_path = Path(file_path)
    
    # Validate file exists
    if not audio_path.exists():
        return {
            "success": False,
            "transcript": "",
            "error": f"Audio file not found: {file_path}",
        }
    
    if not audio_path.is_file():
        return {
            "success": False,
            "transcript": "",
            "error": f"Path is not a file: {file_path}",
        }
    
    # Validate file extension
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported file format: {audio_path.suffix}. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }
    
    # Validate file size
    try:
        file_size = audio_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "transcript": "",
                "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024)}MB)",
            }
    except OSError as e:
        logger.error("Failed to get file size for %s: %s", file_path, e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Failed to access file: {e}",
        }

    # Use provided model, or fall back to default
    if model is None:
        model = DEFAULT_STT_MODEL

    try:
        from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

        client = OpenAI(api_key=api_key, base_url=base_url)

        with open(file_path, "rb") as audio_file:
            response_format = "text" if model == "whisper-1" else "json"
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format=response_format,
            )

        transcript_text = _extract_transcript_text(transcription)

        logger.info("Transcribed %s (%d chars)", audio_path.name, len(transcript_text))

        return {
            "success": True,
            "transcript": transcript_text,
        }

    except PermissionError:
        logger.error("Permission denied accessing file: %s", file_path, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Permission denied: {file_path}",
        }
    except APIConnectionError as e:
        logger.error("API connection error during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Connection error: {e}",
        }
    except APITimeoutError as e:
        logger.error("API timeout during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Request timeout: {e}",
        }
    except APIError as e:
        logger.error("OpenAI API error during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"API error: {e}",
        }
    except Exception as e:
        logger.error("Unexpected error during transcription: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Transcription failed: {e}",
        }


def _resolve_openai_audio_client_config() -> tuple[str, str]:
    """Return direct OpenAI audio credentials or managed gateway fallback."""
    direct_api_key = os.getenv("VOICE_TOOLS_OPENAI_KEY", "").strip()
    if direct_api_key:
        return direct_api_key, "https://api.openai.com/v1"

    managed_gateway = resolve_managed_tool_gateway("openai-audio")
    if managed_gateway is None:
        raise ValueError(
            "VOICE_TOOLS_OPENAI_KEY not set and managed OpenAI audio gateway is unavailable"
        )

    return managed_gateway.nous_user_token, urljoin(
        f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"
    )


def _extract_transcript_text(transcription: Any) -> str:
    """Normalize text and JSON transcription response shapes to a plain string."""
    if isinstance(transcription, str):
        return transcription.strip()

    if hasattr(transcription, "text"):
        value = getattr(transcription, "text")
        if isinstance(value, str):
            return value.strip()

    if isinstance(transcription, dict):
        value = transcription.get("text")
        if isinstance(value, str):
            return value.strip()

    return str(transcription).strip()
