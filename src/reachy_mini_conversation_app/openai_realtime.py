import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.local_audio import LocalASR, LocalTTS, LocalVAD
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates to match OpenAI's requirements
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        # Override type annotations for OpenAI strict typing (only for values used in API)
        self.output_sample_rate = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate = OPEN_AI_INPUT_SAMPLE_RATE

        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.start_time = asyncio.get_event_loop().time()
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0  # sequence counter to prevent stale emissions
        self.partial_debounce_delay = 0.5  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # =====================================================================
        # LOCAL AUDIO COMPONENTS (for full local mode)
        # =====================================================================

        # Built-in Local VAD (always available)
        self._local_vad = LocalVAD(
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration=config.VAD_SILENCE_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            sample_rate=self.input_sample_rate,
        )
        self._audio_buffer: list[bytes] = []  # Buffer for audio during speech
        self._is_speech_active: bool = False
        self._vad_processing: bool = False  # Prevent concurrent processing

        # External VAD endpoint (optional - for smart turn detection)
        self._local_vad_endpoint: str | None = config.LOCAL_VAD_ENDPOINT
        if self._local_vad_endpoint:
            logger.info("External VAD enabled at %s", self._local_vad_endpoint)

        # Built-in Local ASR (Distil-Whisper - lightweight for edge)
        self._local_asr: LocalASR | None = None

        self._local_asr = LocalASR(
            model_name=config.DISTIL_WHISPER_MODEL,
            language=config.WHISPER_LANGUAGE,
        )
        logger.info("Built-in ASR (Distil-Whisper) initialized: %s model", config.DISTIL_WHISPER_MODEL)

        # Built-in Local TTS (Kokoro via FastRTC - lightweight for edge)
        self._local_tts: LocalTTS | None = None

        # Use built-in Kokoro via FastRTC
        self._local_tts = LocalTTS(
            output_sample_rate=self.output_sample_rate,
            voice=config.KOKORO_VOICE,
            speed=config.KOKORO_SPEED,
        )
        logger.info("Built-in TTS initialized: Kokoro via FastRTC (voice: %s)", config.KOKORO_VOICE)

        # =====================================================================
        # LOCAL LLM CLIENT (LM Studio, Ollama, or vLLM)
        # =====================================================================
        self._local_llm_client: AsyncOpenAI | None = None
        self._local_llm_model: str = config.LOCAL_LLM_MODEL or "local-model"
        self._local_llm_provider: str = config.LLM_PROVIDER or "vllm"
        self._conversation_history: list[dict[str, Any]] = []
        self._pending_response_id: str | None = None  # Track OpenAI response to cancel

        if config.LOCAL_LLM_ENDPOINT:
            try:
                # Both LM Studio and Ollama use OpenAI-compatible APIs
                self._local_llm_client = AsyncOpenAI(
                    base_url=config.LOCAL_LLM_ENDPOINT,
                    api_key="not-needed",  # Local LLMs don't require API key
                )
                provider_name = config.LLM_PROVIDER.upper() if config.LLM_PROVIDER else "Local LLM"
                logger.info(
                    "%s client initialized at %s with model %s",
                    provider_name,
                    config.LOCAL_LLM_ENDPOINT,
                    self._local_llm_model,
                )
            except Exception as e:
                logger.error("Failed to initialize local LLM client: %s", e)
                self._local_llm_client = None

        logger.info("=" * 60)
        logger.info("FULL LOCAL MODE: No OpenAI connection required")
        logger.info("=" * 60)

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    def _split_into_chunks(self, text: str, max_chars: int = 150) -> list[str]:
        """Split text into optimal chunks for TTS streaming.

        Uses a waterfall approach inspired by mlx-audio:
        1. First try to split at sentence boundaries (.!?…)
        2. Then try clause boundaries (:;)
        3. Then try phrase boundaries (,—)
        4. Finally fall back to space boundaries

        Args:
                text: The text to split.
                max_chars: Maximum characters per chunk (default 250 for fast response).

        Returns:
                List of text chunks to synthesize separately.

        """
        import re

        text = text.strip()
        if not text:
            return []

        # If text is short enough, return as-is
        if len(text) <= max_chars:
            return [text]

        chunks = []
        remaining = text

        # Waterfall punctuation priorities (strongest to weakest break points)
        waterfall = [
            r"([.!?…]+[\"\'\)]?\s+)",  # Sentence endings (with optional quotes/parens)
            r"([:;]\s+)",  # Clause separators
            r"([,—]\s+)",  # Phrase separators
            r"(\s+)",  # Any whitespace (last resort)
        ]

        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining.strip())
                break

            # Try each punctuation level to find a good break point
            best_break = None
            for pattern in waterfall:
                # Find all matches within the max_chars window
                matches = list(re.finditer(pattern, remaining[: max_chars + 50]))
                if matches:
                    # Take the last match that's within or close to max_chars
                    for match in reversed(matches):
                        if match.end() <= max_chars + 20:  # Allow slight overflow for natural breaks
                            best_break = match.end()
                            break
                    if best_break:
                        break

            if best_break and best_break > 20:  # Don't create tiny chunks
                chunk = remaining[:best_break].strip()
                remaining = remaining[best_break:].strip()
            else:
                # No good break point found, force break at max_chars
                # Try to at least break at a space
                space_idx = remaining[:max_chars].rfind(" ")
                if space_idx > 20:
                    chunk = remaining[:space_idx].strip()
                    remaining = remaining[space_idx:].strip()
                else:
                    chunk = remaining[:max_chars].strip()
                    remaining = remaining[max_chars:].strip()

            if chunk:
                chunks.append(chunk)

        return chunks

    async def _check_turn_complete(self, audio_data: bytes) -> bool:
        """Check if the user's turn is complete using local VAD.

        Args:
                audio_data: Raw PCM audio bytes (16-bit, 24kHz, mono)

        Returns:
                True if turn is complete, False if user might still be speaking

        """
        if not self._local_vad_endpoint:
            return True  # No VAD configured, assume complete

        try:
            import wave
            import tempfile

            import aiohttp

            # Save audio to temp WAV file for the VAD server
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, "wb") as wav:
                    wav.setnchannels(1)  # mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(self.input_sample_rate)  # 24kHz
                    wav.writeframes(audio_data)

            # Read the WAV file and encode as base64
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()

            import base64

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Clean up temp file
            try:
                import os

                os.unlink(temp_path)
            except Exception:
                pass

            # Call VAD endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._local_vad_endpoint}/predict",
                    json={"audio_base64": audio_b64},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        prediction = result.get("prediction", 1)
                        probability = result.get("probability", 1.0)
                        status = result.get("status", "complete")
                        logger.info("VAD result: %s (probability=%.2f)", status, probability)
                        return prediction == 1  # 1 = complete, 0 = incomplete
                    else:
                        logger.warning("VAD request failed with status %d", resp.status)
                        return True  # Assume complete on error

        except Exception as e:
            logger.error("VAD check failed: %s", e)
            return True  # Assume complete on error

    async def _transcribe_with_local_asr(self, audio_data: bytes) -> str | None:
        """Transcribe audio using local ASR (Distil-Whisper).

        Args:
                audio_data: Raw PCM audio bytes (16-bit, 24kHz, mono)

        Returns:
                Transcribed text or None if failed

        """
        # Use built-in Distil-Whisper
        if self._local_asr:
            try:
                transcript = await self._local_asr.transcribe(audio_data, self.input_sample_rate)
                if transcript:
                    return transcript
                logger.warning("Built-in ASR returned empty result")
                return None
            except Exception as e:
                logger.error("Built-in ASR transcription failed: %s", e)
                return None

    async def _process_local_asr(self, audio_data: bytes, check_vad: bool = True) -> None:
        """Process audio with local ASR and generate response with local LLM.

        Args:
                audio_data: Raw PCM audio bytes from the speech buffer.
                check_vad: Whether to check VAD first (default True).

        """
        # Check if turn is complete using local VAD
        if check_vad and self._local_vad_endpoint:
            is_complete = await self._check_turn_complete(audio_data)
            if not is_complete:
                logger.info("VAD says turn incomplete - waiting for more speech")
                # Re-enable speech buffering to capture more audio
                self._is_speech_active = True
                # Put the audio back in the buffer
                self._audio_buffer.append(audio_data)
                return

        # Transcribe with local ASR
        transcript = await self._transcribe_with_local_asr(audio_data)
        if not transcript:
            logger.warning("Local ASR returned no transcription")
            return

        # Show transcription in UI
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        # Generate response with local LLM
        if self._local_llm_client:
            await self._generate_local_response(transcript)
        else:
            logger.warning("Local LLM not available, cannot generate response")

    async def _generate_local_response(self, user_message: str) -> None:
        """Generate a response using the local LLM and send to Chatterbox.

        Args:
                user_message: The user's transcribed message.

        """
        if not self._local_llm_client:
            logger.warning("Local LLM client not available")
            return

        try:
            # Add user message to conversation history
            self._conversation_history.append({"role": "user", "content": user_message})

            # Build messages with system prompt
            messages = [
                {"role": "system", "content": get_session_instructions()},
                *self._conversation_history[-20:],  # Keep last 20 messages for context
            ]

            logger.debug("Calling local LLM with %d messages", len(messages))

            # Call local LLM (no tool support - using base instruct model)
            response = await self._local_llm_client.chat.completions.create(
                model=self._local_llm_model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                tools=get_tool_specs(),
            )

            choice = response.choices[0]
            assistant_message = choice.message

            logger.info("Received tool calls from LLM: %s", assistant_message.tool_calls)
            if assistant_message.tool_calls:
                logger.info("Received tool calls from LLM: %s", assistant_message.tool_calls)
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    logger.debug("Executing tool: %s with args: %s", tool_name, args_str)

                    try:
                        tool_result = await dispatch_tool_call(tool_name, args_str, self.deps)
                        logger.debug("Tool '%s' result: %s", tool_name, tool_result)
                    except Exception as e:
                        logger.error("Tool '%s' failed: %s", tool_name, e)
                        tool_result = {"error": str(e)}

                    tool_results.append(tool_result)

                # Now send the tool results back to the LLM for a final response
                # You may need to add these results as a new message to conversation history
                # But the current approach might not work with Ollama natively unless it supports
                # multi-turn tool calling properly.

                # For now, just log the results and proceed
                logger.info("Tool results: %s", tool_results)

            # Get the text content
            text_response = assistant_message.content
            if text_response:
                # Clean up thinking tags from various models (Qwen, DeepSeek, etc.)
                import re

                # Remove <think>...</think> tags (Qwen style)
                if "<think>" in text_response:
                    text_response = re.sub(r"<think>.*?</think>", "", text_response, flags=re.DOTALL).strip()
                # Remove <thinking>...</thinking> tags (other models)
                if "<thinking>" in text_response:
                    text_response = re.sub(r"<thinking>.*?</thinking>", "", text_response, flags=re.DOTALL).strip()

                logger.info("Local LLM response: %s", text_response[:100])

                # Add to conversation history
                self._conversation_history.append({"role": "assistant", "content": text_response})

                # Show in UI
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text_response}))

                # Synthesize with local TTS
                await self._synthesize_locally(text_response)

        except Exception as e:
            logger.error("Local LLM generation failed: %s", e)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[error] LLM failed: {e}"})
            )

    async def _synthesize_locally(self, text: str) -> None:
        """Synthesize text using the configured local TTS provider.

        Args:
                text: The text to synthesize.

        """
        if not text or not text.strip():
            return

        # Use built-in local TTS (Kokoro via FastRTC)
        if self._local_tts:
            try:
                audio_data = await self._local_tts.synthesize(text)
                if audio_data is not None:
                    # Feed to head wobbler if available
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.feed(base64.b64encode(audio_data.tobytes()).decode("utf-8"))

                    # Queue audio in chunks for smoother playback
                    chunk_size = 4800  # 200ms at 24kHz
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i : i + chunk_size]
                        await self.output_queue.put(
                            (self.output_sample_rate, chunk.reshape(1, -1)),
                        )
                    logger.debug("Local TTS synthesis complete")
                else:
                    logger.warning("Local TTS returned no audio")
            except Exception as e:
                logger.error("Local TTS synthesis failed: %s", e)
        else:
            logger.warning("No TTS provider available")

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions()
                voice = get_session_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session={
                            "type": "realtime",
                            "instructions": instructions,
                            "audio": {"output": {"voice": voice}},
                        },
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, sequence: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)
            # Only emit if this is still the latest partial (by sequence number)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        await self._run_local_only_session()
        return

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: OpenAI client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            asyncio.create_task(self._run_realtime_session(), name="openai-realtime-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _run_local_only_session(self) -> None:
        """Run in full local mode without any OpenAI connection.

        This handles the entire pipeline locally:
        - Energy-based VAD for speech start detection
        - Smart-turn VAD for turn completion
        - Local ASR (GLM-ASR-Nano)
        - Local LLM (Qwen via vLLM)
        - Local TTS (Chatterbox)
        """
        logger.info("Local-only session started - VAD, ASR, LLM, and TTS all running locally")

        # Signal that we're ready to receive audio
        self._connected_event.set()

        # The audio processing happens in receive() which is called by the audio input stream
        # We just need to keep this session alive
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

        logger.info("Local-only session ended")

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
            try:
                # Build session config - conditionally include audio output based on Chatterbox
                audio_config: dict[str, Any] = {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.input_sample_rate,
                        },
                        "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                        "turn_detection": {
                            "type": "server_vad",
                            "interrupt_response": True,
                        },
                    },
                }

                audio_config["output"] = {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.output_sample_rate,
                    },
                    "voice": get_session_voice(),
                }

                session_config: dict[str, Any] = {
                    "type": "realtime",
                    "instructions": get_session_instructions(),
                    "audio": audio_config,
                    "tools": get_tool_specs(),
                    "tool_choice": "auto",
                }

                # When using local LLM, configure OpenAI for transcription only (no response generation)
                if self._local_llm_client:
                    # Minimal instructions since we won't use OpenAI's responses
                    session_config["instructions"] = "You are a transcription service. Do not respond."
                    # No tools - local LLM handles tool calls
                    session_config["tools"] = []
                    session_config["tool_choice"] = "none"
                    # Remove audio output config - we don't want OpenAI to speak
                    if "output" in session_config.get("audio", {}):
                        del session_config["audio"]["output"]
                    logger.info(
                        "Local LLM enabled - OpenAI transcription-only mode, local LLM will generate responses"
                    )

                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    get_session_voice(),
                )
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                return

            logger.info("Realtime session updated successfully")

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and process it.

        In full local mode, audio is processed entirely locally (VAD, ASR, LLM, TTS).
        Otherwise, audio is sent to the OpenAI server for processing.

        Handles both mono and stereo audio formats, converting to the expected
        mono format. Resamples if the input sample rate differs from the expected rate.

        Args:
                frame: A tuple containing (sample_rate, audio_data).

        """
        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Full local mode: use built-in VAD + ASR + LLM + TTS
        # Process with built-in VAD
        speech_started, speech_ended = self._local_vad.process(audio_frame)

        if speech_started:
            self._is_speech_active = True
            self._audio_buffer.clear()
            self.deps.movement_manager.set_listening(True)
            logger.info("VAD: speech started")

        if self._is_speech_active:
            self._audio_buffer.append(audio_frame.tobytes())

        if speech_ended and not self._vad_processing:
            self._vad_processing = True
            self._is_speech_active = False
            self.deps.movement_manager.set_listening(False)

            audio_data = b"".join(self._audio_buffer)
            self._audio_buffer.clear()
            logger.info("VAD: speech ended (%d bytes)", len(audio_data))

            # Process in background (ASR -> LLM -> TTS)
            asyncio.create_task(self._process_local_speech(audio_data))

    async def _process_local_speech(self, audio_data: bytes) -> None:
        """Process speech audio: ASR -> LLM -> TTS.

        Args:
                audio_data: Raw PCM audio bytes from the speech buffer.

        """
        try:
            # Transcribe with local ASR
            transcript = await self._transcribe_with_local_asr(audio_data)
            if not transcript:
                logger.warning("ASR returned no transcription")
                return

            # Show transcription in UI
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

            # Generate response with local LLM
            if self._local_llm_client:
                await self._generate_local_response(transcript)
            else:
                logger.warning("Local LLM not available, cannot generate response")

        finally:
            self._vad_processing = False

    async def _process_with_local_vad(self, audio_data: bytes) -> None:
        """Process audio with local VAD check then ASR/LLM if turn complete."""
        try:
            is_complete = await self._check_turn_complete(audio_data)

            if is_complete:
                logger.info("Local VAD: turn complete, proceeding to ASR")
                self._is_speech_active = False
                self._vad_speech_frames = 0
                self._audio_buffer.clear()
                self.deps.movement_manager.set_listening(False)

                # Process with ASR and LLM (skip VAD check since we just did it)
                await self._process_local_asr(audio_data, check_vad=False)
            else:
                logger.info("Local VAD: turn incomplete, continuing to listen")
                # Keep listening - don't clear buffer, just reset silence counter
                self._vad_silence_frames = 0
        finally:
            self._vad_processing = False

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"
