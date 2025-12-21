import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

# Locate .env file (search upward from current working directory)
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load .env and override environment variables
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    # Chatterbox TTS configuration (optional - when set, uses Chatterbox instead of OpenAI TTS)
    CHATTERBOX_ENDPOINT = os.getenv("CHATTERBOX_ENDPOINT")  # e.g., "http://192.168.68.74:7860"
    CHATTERBOX_REF_AUDIO = os.getenv("CHATTERBOX_REF_AUDIO")  # Optional reference audio path for voice cloning
    if CHATTERBOX_ENDPOINT:
        logger.info(f"Chatterbox TTS enabled at {CHATTERBOX_ENDPOINT}")

    # Local LLM configuration (optional - when set, uses local LLM instead of OpenAI for conversation)
    LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT")  # e.g., "http://192.168.68.74:8002/v1"
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen3-30B")  # Model name served by vLLM
    if LOCAL_LLM_ENDPOINT:
        logger.info(f"Local LLM enabled at {LOCAL_LLM_ENDPOINT} with model {LOCAL_LLM_MODEL}")

    # Local ASR configuration (optional - when set, uses local ASR instead of OpenAI for speech-to-text)
    LOCAL_ASR_ENDPOINT = os.getenv("LOCAL_ASR_ENDPOINT")  # e.g., "http://192.168.68.74:7862"
    if LOCAL_ASR_ENDPOINT:
        logger.info(f"Local ASR enabled at {LOCAL_ASR_ENDPOINT}")

    # Local VAD configuration (optional - smart turn detection to replace OpenAI VAD)
    LOCAL_VAD_ENDPOINT = os.getenv("LOCAL_VAD_ENDPOINT")  # e.g., "http://192.168.68.74:7863"
    if LOCAL_VAD_ENDPOINT:
        logger.info(f"Local VAD enabled at {LOCAL_VAD_ENDPOINT}")


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
