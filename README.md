**# Reachy Mini Conversation App

**Fully local conversational AI for Reachy Mini robot** - combining lightweight speech recognition, text-to-speech, and local LLM with choreographed motion libraries.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Features

- 🎯 **100% Local Operation** - No cloud dependencies, runs entirely on-device
- 🎤 **Real-time Audio** - Low-latency speech-to-text (Distil-Whisper) and text-to-speech (Kokoro)
- 🤖 **Local LLM** - Powered by Ollama or LM Studio for on-device conversation
- 💃 **Motion System** - Layered motion with dances, emotions, face-tracking, and speech-reactive movement
- 🎨 **Custom Personalities** - Easy profile system for different robot behaviors
- 🔧 **Edge-Optimized** - Designed for Jetson Nano and similar edge devices

## Prerequisites

> [!IMPORTANT]
> **Install Reachy Mini SDK first**: [github.com/pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini/)
>
> Works with:
> - **Real hardware** - Physical Reachy Mini robot
> - **Simulator** - Virtual Reachy Mini for testing

## Quick Start

### 1. Install the App

```bash
# Clone repository
git clone <repo-url>
cd reachy_mini_conversation_app

# Install dependencies
pip install -e "."

# For Jetson Nano with CUDA optimization:
pip install -e ".[jetson]"
```

### 2. Install Local LLM

**Ollama (Recommended):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi-3-mini-4k-instruct
```

**Or LM Studio:**
- Download from [lmstudio.ai](https://lmstudio.ai)
- Load a GGUF model (e.g., Phi-3-mini)
- Start local server on port 1234

### 3. Configure

```bash
# Copy example config
cp .env.example .env

# Edit if needed (defaults work for most setups)
nano .env
```

### 4. Run

**Console mode (headless):**
```bash
reachy-mini-conversation-app
```

**Web UI mode (required for simulator):**
```bash
reachy-mini-conversation-app --gradio
```

Access at `http://localhost:7860`

## Configuration

The app auto-configures for your hardware. Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama` or `lmstudio`) |
| `OLLAMA_MODEL` | `phi-3-mini-4k-instruct` | Ollama model name |
| `DISTIL_WHISPER_MODEL` | `distil-small.en` | Speech recognition model |
| `KOKORO_VOICE` | `af_sarah` | TTS voice (af_sarah, am_michael, etc.) |
| `JETSON_OPTIMIZE` | `true` | Enable Jetson-specific optimizations |

See `.env.jetson` for Jetson Nano optimized settings.

## CLI Options

| Option | Description |
|--------|-------------|
| `--gradio` | Launch web UI (required for simulator) |
| `--head-tracker {yolo,mediapipe}` | Enable face tracking |
| `--local-vision` | Use local vision model (requires `local_vision` extra) |
| `--no-camera` | Disable camera (audio-only mode) |
| `--wireless-version` | Use GStreamer for wireless robots |
| `--debug` | Enable verbose logging |

## Optional Extras

```bash
# Vision features
pip install -e ".[local_vision]"      # Local vision model (SmolVLM2)
pip install -e ".[yolo_vision]"       # YOLO face tracking
pip install -e ".[mediapipe_vision]"  # MediaPipe tracking
pip install -e ".[all_vision]"        # All vision features

# Hardware support
pip install -e ".[reachy_mini_wireless]"  # Wireless Reachy Mini
pip install -e ".[jetson]"                 # Jetson optimization (CUDA)

# Development
pip install -e ".[dev]"  # Testing & linting tools
```

## Available Tools

The LLM has access to these robot actions:

| Tool | Action |
|------|--------|
| `move_head` | Move head (left/right/up/down/front) |
| `camera` | Capture and analyze camera image |
| `head_tracking` | Enable/disable face tracking |
| `dance` | Play choreographed dance |
| `stop_dance` | Stop current dance |
| `play_emotion` | Display emotion animation |
| `stop_emotion` | Stop emotion animation |
| `do_nothing` | Remain idle |

## Custom Personalities

Create custom robot personalities with unique behaviors:

1. Set profile name: `REACHY_MINI_CUSTOM_PROFILE=my_profile` in `.env`
2. Create folder: `src/reachy_mini_conversation_app/profiles/my_profile/`
3. Add files:
   - `instructions.txt` - Personality prompt
   - `tools.txt` - Available tools (one per line)
   - `custom_tool.py` - Optional custom tools

See `profiles/example/` for reference.

**Live editing with Gradio UI:**
- Use the "Personality" panel to switch profiles
- Create new personalities directly from the UI
- Changes apply immediately to current session


**Expected performance:**
- End-to-end latency: <3 seconds
- Memory usage: ~3GB peak
- Fully offline operation

## Troubleshooting

**TimeoutError connecting to robot:**
```bash
# Start the Reachy Mini daemon first
# See: https://github.com/pollen-robotics/reachy_mini/
```

**No audio output:**
- Check TTS voice is valid: `af_sarah`, `am_michael`, `bf_emma`, `bm_lewis`
- Verify Ollama/LM Studio is running: `curl http://localhost:11434` or `:1234`

**Out of memory (Jetson):**
- Use smaller model: `OLLAMA_MODEL=llama3.2:1b`
- Disable vision: `--no-camera`

## Architecture

```
User Speech → VAD → Distil-Whisper STT → Local LLM → Kokoro TTS → Audio Output
                                              ↓
                                         Tool Dispatch
                                              ↓
                                    Robot Actions (Motion/Vision)
```

All processing runs locally using:
- **VAD**: Built-in energy-based detection
- **STT**: Distil-Whisper (lightweight, 2-6x faster)
- **LLM**: Ollama/LM Studio (Phi-3-mini recommended)
- **TTS**: Kokoro-82M via FastRTC (production quality)
- **Framework**: FastRTC for low-latency audio streaming

## Development

```bash
# Install dev tools
pip install -e ".[dev]"

# Run linter
ruff check .

# Run tests
pytest
```

## License

Apache 2.0

---

**Built for edge deployment** - Optimized for any hardware with 8GB+ RAM.
**
** Thanks to muellerzr for his fork **
