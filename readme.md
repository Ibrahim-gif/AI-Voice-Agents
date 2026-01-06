# Voice Assistant with Function Calling (Azure VoiceLive + PyAudio)

A real-time, bidirectional voice assistant built with the **Azure VoiceLive SDK**.  
It captures microphone audio (24kHz PCM16 mono), streams it to VoiceLive over WebSockets, plays back synthesized audio responses, and supports **tool/function calling** to:

- **upsert_person_profile**: store structured client intake fields (patch semantics)
- **end_conversation**: cleanly end the session once intake is complete

This project is designed as a **live call intake agent**: greet → collect a small set of fields → confirm → end the call.

---

## Features

- 🎙️ **Real-time streaming audio** (mic → VoiceLive → speakers)
- 🧠 **Function calling** integrated with VoiceLive tools
- 🛑 **Barge-in support**: cancels assistant audio if the user starts speaking
- 📝 **Automatic logging** to timestamped log files in `./logs/`
- 🔐 Auth via:
  - **API Key** (`AzureKeyCredential`)
  - **Azure token credential** (`AzureCliCredential` / `DefaultAzureCredential` pattern)

# Installation

## 1. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

## 2. Install dependencies
```
pip install -r requirements.txt
```

## 3. Configuration (.env)

Create a .env file in the same folder as the script:

```
# Required if using API key authentication
AZURE_VOICELIVE_API_KEY=YOUR_KEY_HERE

# Required (your resource endpoint)
AZURE_VOICELIVE_ENDPOINT=https://your-resource-name.services.ai.azure.com/

# Optional
AZURE_VOICELIVE_MODEL=gpt-realtime
AZURE_VOICELIVE_VOICE=en-US-Ava:DragonHDLatestNeural

# Optional: system prompt / instructions (can be large)
AZURE_VOICELIVE_INSTRUCTIONS="You are a real-time voice intake agent..."\
```

## 4. Run 

```
python azure-realtime-audio.py
```