# Reachy Mini Conversation App with Vision

**Fully local conversational AI for Reachy Mini robot** - Combining lightweight speech recognition, text-to-speech, vision and local LLM with choreographed motion libraries.

## Features

- 🎯 **100% Local Operation** - No cloud dependencies, runs entirely on-device
- 🎤 **Real-time Audio** - Low-latency speech-to-text (Distil-Whisper) and text-to-speech (Kokoro)
- 🤖 **Local LLM** - Powered by Ollama or LM Studio for on-device conversation
- 👁️ **Local Vision** - Powered by HuggingFaceTB/SmolVLM2
- 💃 **Motion System** - Layered motion with dances, emotions, face-tracking, and speech-reactive movement
- 🎨 **Custom Personalities** - Easy profile system for different robot behaviors
- 🔧 **Edge-Optimized** - Designed for Jetson Nano and similar edge devices

## Prerequisites

> [!IMPORTANT]
> **Learn how to use and setup Reachy Mini first!**: [Getting Started](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/get_started) - [ReachyMini Repo](https://github.com/pollen-robotics/reachy_mini/)
> 
> For Wireless Reachy, you will also need Gstreamer installed via [Homebrew](https://brew.sh)
> ```
> brew install gstreamer libnice-gstreamer
> ```
> 
> Works with:
> - **Real hardware** - Physical Reachy Mini robot
> - **Simulator** - Virtual Reachy Mini for testing

## Quick Start

### 1. Install the App

```bash
# Clone repository
git clone <repo-url>
cd reachy_mini_conversation_app_local

# Setup python VENV
python -m venv reachy-mini-env
source reachy-mini-env/bin/activate

# Install Requirements
pip install -r requirements.txt

# Install project
pip install -e "."

# OR - For Jetson Nano with CUDA optimization:
pip install -e ".[jetson]"
```

### 2. Install Local LLM

**Ollama (Recommended):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:1b
```

**Or LM Studio:**
- Download from [lmstudio.ai](https://lmstudio.ai)
- Load a GGUF model (e.g., Phi-3-mini)
- Start local server on port 1234

### 3. Configure

```bash
# Copy example config
cp .env.example .env

# Edit .env
# You will need a HuggingFace API token to download local models. Add it to the HF_TOKEN in .env
nano .env
```

### 4. Run

NOTE: (Wireless Reachy) Make sure you "wake up" reachy via the app or [http://reachy-mini.local:8000](http://reachy-mini.local:8000) 

**Console mode (headless):**
```bash
reachy-mini-conversation-app
```

**Web UI mode (required for simulator):**
```bash
reachy-mini-conversation-app --gradio
```

Access gradio site at `http://localhost:7860`

## Configuration

The app auto-configures for your hardware. Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama` or `lmstudio`) |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Ollama model name |
| `DISTIL_WHISPER_MODEL` | `distil-small.en` | Speech recognition model |
| `KOKORO_VOICE` | `af_sarah` | TTS voice (af_sarah, am_michael, etc.) |
| `CUDA_SUPPORT` | `false` | If running on NVIDIA hardware |
| `LOCAL_VISION_MODEL` | `SmolVLM2-2.2B-Instruct` | Vision recognition model |

See `.env.example` for more, and `.env.jetson` for Jetson Nano optimized settings.

## CLI Options

| Option | Description |
|--------|-------------|
| `--gradio` | Launch web UI (required for simulator) |
| `--no-camera` | Disable all camera usage (audio-only mode) |
| `--no-vision` | Disable vision model processing |
| `--no-head-tracking` | Disable head tracking |
| `--head-tracker mediapipe` | Use mediapipe instead of yolo |
| `--debug` | Enable verbose logging |

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

**Thanks to muellerzr and dwain-barnes for their forks**
