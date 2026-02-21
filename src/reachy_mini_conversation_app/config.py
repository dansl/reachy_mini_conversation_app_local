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
	OPENAI_API_KEY = ""

    """Configuration class for the conversation app."""

    # =========================================================================
    # FULL LOCAL MODE (ALWAYS ENABLED - No cloud dependencies)
    # =========================================================================
    FULL_LOCAL_MODE = True  # Hardcoded for fully local operation

    # =========================================================================
    # JETSON OPTIMIZATION
    # =========================================================================
    JETSON_OPTIMIZE = os.getenv("JETSON_OPTIMIZE", "true").lower().strip() in ("true", "1", "yes")

    # =========================================================================
    # VISION CONFIGURATION
    # =========================================================================
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    # =========================================================================
    # LOCAL LLM CONFIGURATION (Required for fully local operation)
    # =========================================================================
    # Default to Ollama for Jetson optimization
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama" if JETSON_OPTIMIZE else "").lower().strip()

    # LM Studio configuration
    LMSTUDIO_ENDPOINT = os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234/v1")
    LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "")

    # Ollama configuration (recommended for Jetson)
    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/v1")
    # Default to phi-3-mini for Jetson (3.8B params, ~2GB RAM)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi-3-mini-4k-instruct" if JETSON_OPTIMIZE else "")

    # Resolve LOCAL_LLM_ENDPOINT and LOCAL_LLM_MODEL based on provider selection
    LOCAL_LLM_ENDPOINT: str | None = None
    LOCAL_LLM_MODEL: str = ""

    if LLM_PROVIDER == "lmstudio":
        LOCAL_LLM_ENDPOINT = LMSTUDIO_ENDPOINT
        LOCAL_LLM_MODEL = LMSTUDIO_MODEL or "local-model"
        logger.info(f"LM Studio enabled at {LOCAL_LLM_ENDPOINT} with model {LOCAL_LLM_MODEL}")
    elif LLM_PROVIDER == "ollama":
        LOCAL_LLM_ENDPOINT = OLLAMA_ENDPOINT
        LOCAL_LLM_MODEL = OLLAMA_MODEL or "llama3"
        logger.info(f"Ollama enabled at {LOCAL_LLM_ENDPOINT} with model {LOCAL_LLM_MODEL}")
    elif LLM_PROVIDER:
        logger.warning(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Valid options: 'lmstudio', 'ollama'")

    # Legacy support: allow direct LOCAL_LLM_ENDPOINT override
    _legacy_endpoint = os.getenv("LOCAL_LLM_ENDPOINT")
    _legacy_model = os.getenv("LOCAL_LLM_MODEL")
    if _legacy_endpoint and not LLM_PROVIDER:
        LOCAL_LLM_ENDPOINT = _legacy_endpoint
        LOCAL_LLM_MODEL = _legacy_model or "Qwen3-30B"
        logger.info(f"Local LLM enabled at {LOCAL_LLM_ENDPOINT} with model {LOCAL_LLM_MODEL} (legacy config)")

    # =========================================================================
    # LOCAL ASR CONFIGURATION (Speech-to-Text - Distil-Whisper)
    # =========================================================================
    # Distil-Whisper model selection (optimized for Jetson)
    DISTIL_WHISPER_MODEL = os.getenv(
        "DISTIL_WHISPER_MODEL",
        "distil-whisper/distil-small.en" if JETSON_OPTIMIZE else "distil-whisper/distil-medium.en"
    )
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    logger.info(f"Local ASR: Distil-Whisper ({DISTIL_WHISPER_MODEL}, {WHISPER_LANGUAGE})")

    # =========================================================================
    # LOCAL TTS CONFIGURATION (Text-to-Speech - Kokoro via FastRTC)
    # =========================================================================
    # Kokoro settings (lightweight, 82M parameters, via FastRTC)
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")  # Voice selection
    KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))  # Speech speed (0.5-2.0)

    logger.info(f"Local TTS: Kokoro via FastRTC (voice: {KOKORO_VOICE}, speed: {KOKORO_SPEED})")

    # =========================================================================
    # LOCAL VAD CONFIGURATION (Voice Activity Detection)
    # =========================================================================
    # VAD is built-in (energy-based). External endpoint is optional.
    VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.01"))
    VAD_SILENCE_DURATION = float(os.getenv("VAD_SILENCE_DURATION", "0.8"))
    VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.3"))

    # External VAD endpoint (optional - for smart turn detection)
    LOCAL_VAD_ENDPOINT = os.getenv("LOCAL_VAD_ENDPOINT")  # e.g., "http://192.168.68.74:7863"
    if LOCAL_VAD_ENDPOINT:
        logger.info(f"External VAD enabled at {LOCAL_VAD_ENDPOINT}")

    # =========================================================================
    # ONNX RUNTIME OPTIMIZATION (for Jetson)
    # =========================================================================
    ONNX_PROVIDERS = os.getenv(
        "ONNX_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider" if JETSON_OPTIMIZE else "CPUExecutionProvider"
    )

    # =========================================================================
    # SYSTEM STATUS LOGGING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("FULLY LOCAL MODE - No cloud dependencies")
    if JETSON_OPTIMIZE:
        logger.info("Jetson Optimization: ENABLED")
        logger.info(f"Recommended LLM: {OLLAMA_MODEL}")
    logger.info("=" * 60)


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
