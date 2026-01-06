# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations
import os
import sys
import argparse
import asyncio
import json
import base64
from datetime import datetime
import logging
import queue
import signal
from typing import Union, Optional, Dict, Any, Mapping, Callable, TYPE_CHECKING, cast

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureCliCredential, DefaultAzureCredential

from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AudioEchoCancellation,
    AudioNoiseReduction,
    AzureStandardVoice,
    InputAudioFormat,
    ItemType,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    ServerVad,
    FunctionTool,
    FunctionCallOutputItem,
    ToolChoiceLiteral,
    AudioInputTranscriptionOptions,
    Tool,
)
from dotenv import load_dotenv
import pyaudio

if TYPE_CHECKING:
    from azure.ai.voicelive.aio import VoiceLiveConnection

## Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Environment variable loading
load_dotenv('./.env', override=True)

# Set up logging
## Add folder for logging
if not os.path.exists('logs'):
    os.makedirs('logs')

## Add timestamp for logfiles
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

## Set up logging
logging.basicConfig(
    filename=f'logs/{timestamp}_voicelive.log',
    filemode="w",
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles real-time audio capture and playback for the voice assistant.

    Threading Architecture:
    - Main thread: Event loop and UI
    - Capture thread: PyAudio input stream reading
    - Send thread: Async audio data transmission to VoiceLive
    - Playback thread: PyAudio output stream writing
    """
    
    loop: asyncio.AbstractEventLoop
    
    class AudioPlaybackPacket:
        """Represents a packet that can be sent to the audio playback queue."""
        def __init__(self, seq_num: int, data: Optional[bytes]):
            self.seq_num = seq_num
            self.data = data

    def __init__(self, connection):
        self.connection = connection
        self.audio = pyaudio.PyAudio()

        # Audio configuration - PCM16, 24kHz, mono as specified
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        self.chunk_size = 1200 # 50ms

        # Capture and playback state
        self.input_stream = None

        self.playback_queue: queue.Queue[AudioProcessor.AudioPlaybackPacket] = queue.Queue()
        self.playback_base = 0
        self.next_seq_num = 0
        self.output_stream: Optional[pyaudio.Stream] = None

        logger.info("AudioProcessor initialized with 24kHz PCM16 mono audio")

    def start_capture(self):
        """Start capturing audio from microphone."""
        def _capture_callback(
            in_data,      # data
            _frame_count,  # number of frames
            _time_info,    # dictionary
            _status_flags):
            """Audio capture thread - runs in background."""
            audio_base64 = base64.b64encode(in_data).decode("utf-8")
            asyncio.run_coroutine_threadsafe(
                self.connection.input_audio_buffer.append(audio=audio_base64), self.loop
            )
            return (None, pyaudio.paContinue)

        if self.input_stream:
            return

        # Store the current event loop for use in threads
        self.loop = asyncio.get_event_loop()

        try:
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_capture_callback,
            )
            logger.info("Started audio capture")

        except Exception:
            logger.exception("Failed to start audio capture")
            raise

    def start_playback(self):
        """Initialize audio playback system."""
        if self.output_stream:
            return

        remaining = bytes()
        def _playback_callback(
            _in_data,
            frame_count,  # number of frames
            _time_info,
            _status_flags):

            nonlocal remaining
            frame_count *= pyaudio.get_sample_size(pyaudio.paInt16)

            out = remaining[:frame_count]
            remaining = remaining[frame_count:]

            while len(out) < frame_count:
                try:
                    packet = self.playback_queue.get_nowait()
                except queue.Empty:
                    out = out + bytes(frame_count - len(out))
                    continue
                except Exception:
                    logger.exception("Error in audio playback")
                    raise

                if not packet or not packet.data:
                    # None packet indicates end of stream
                    logger.info("End of playback queue.")
                    break

                if packet.seq_num < self.playback_base:
                    # skip requested
                    # ignore skipped packet and clear remaining
                    if len(remaining) > 0:
                        remaining = bytes()
                    continue

                num_to_take = frame_count - len(out)
                out = out + packet.data[:num_to_take]
                remaining = packet.data[num_to_take:]

            if len(out) >= frame_count:
                return (out, pyaudio.paContinue)
            else:
                return (out, pyaudio.paComplete)

        try:
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_playback_callback
            )
            logger.info("Audio playback system ready")
        except Exception:
            logger.exception("Failed to initialize audio playback")
            raise

    def _get_and_increase_seq_num(self):
        seq = self.next_seq_num
        self.next_seq_num += 1
        return seq

    def queue_audio(self, audio_data: Optional[bytes]) -> None:
        """Queue audio data for playback."""
        self.playback_queue.put(
            AudioProcessor.AudioPlaybackPacket(
                seq_num=self._get_and_increase_seq_num(),
                data=audio_data))

    def skip_pending_audio(self):
        """Skip current audio in playback queue."""
        self.playback_base = self._get_and_increase_seq_num()

    def shutdown(self):
        """Clean up audio resources."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

        logger.info("Stopped audio capture")

        # Inform thread to complete
        if self.output_stream:
            self.skip_pending_audio()
            self.queue_audio(None)
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None

        logger.info("Stopped audio playback")

        if self.audio:
            self.audio.terminate()

        logger.info("Audio processor cleaned up")



class AsyncFunctionCallingClient:
    """Voice assistant with function calling capabilities using VoiceLive SDK patterns."""

    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, AsyncTokenCredential],
        model: str,
        voice: str,
        instructions: str,
    ):
        self.endpoint = endpoint
        self.credential = credential
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.session_ready = False
        self.conversation_started = False
        self._active_response = False
        self._response_api_done = False
        self._pending_function_call: Optional[Dict[str, Any]] = None
        self._stop_event = asyncio.Event()
        self.person_profile: Dict[str, Any] = {}
        self.shutdown = False
        

        # Define available functions
        self.available_functions: Dict[str, Callable[[Union[str, Mapping[str, Any]]], Mapping[str, Any]]] = {
            "upsert_person_profile": self.upsert_person_profile,
            "end_conversation": self.end_conversation,
        }

    async def start(self):
        """Start the voice assistant session."""
        try:
            logger.info("Connecting to VoiceLive API with model %s", self.model)

            # Connect to VoiceLive WebSocket API
            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                model=self.model,
            ) as connection:
                conn = connection
                self.connection = conn

                # Initialize audio processor
                ap = AudioProcessor(conn)
                self.audio_processor = ap

                # Configure session for voice conversation
                await self._setup_session()

                # Start audio systems
                ap.start_playback()

                logger.info("Voice assistant with function calling ready! Start speaking...")

                # Process events
                await self._process_events()
        finally:
            if self.audio_processor:
                self.audio_processor.shutdown()

    async def _setup_session(self):
        """Configure the VoiceLive session for audio conversation with function tools."""
        logger.info("Setting up voice conversation session with function tools...")

        # Create voice configuration
        voice_config: Union[AzureStandardVoice, str]
        if self.voice.startswith("en-US-") or self.voice.startswith("en-CA-") or "-" in self.voice:
            # Azure voice
            voice_config = AzureStandardVoice(name=self.voice)
        else:
            # OpenAI voice (alloy, echo, fable, onyx, nova, shimmer)
            voice_config = self.voice

        # Create turn detection configuration
        turn_detection_config = ServerVad(
            threshold=0.5,
            prefix_padding_ms=300,
            silence_duration_ms=500)

        # Define function tools
        function_tools: list[Tool] = [
            FunctionTool(
                name="upsert_person_profile",
                description="Store structured personal profile info (only after consent). Use when the user shares personal details.",
                parameters={
                    "type": "object",
                    "properties": {
                        "person": {
                            "type": "object",
                            "properties": {
                                "full_name": {"type": "string"},
                                "preferred_name": {"type": "string"},
                                "email": {"type": "string"},
                                "phone": {"type": "string"},
                                "timezone": {"type": "string"},
                                "location": {"type": "string"},
                                "company": {"type": "string"},
                                "role": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        "source_utterance": {"type": "string", "description": "Short quote or paraphrase of what the user said."},
                    },
                    "required": ["person"],
                    "additionalProperties": False,
                },
            ),
            FunctionTool(
                name="end_conversation",
                description="End the call/session immediately once intake is complete or the assistant believes it has finished its job.",
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            ),
        ]

        # Create session configuration with function tools
        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=self.instructions,
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            input_audio_echo_cancellation=AudioEchoCancellation(),
            input_audio_noise_reduction=AudioNoiseReduction(type="azure_deep_noise_suppression"),
            tools=function_tools,
            tool_choice=ToolChoiceLiteral.AUTO,
            input_audio_transcription=AudioInputTranscriptionOptions(model="whisper-1"),
        )

        conn = self.connection
        assert conn is not None, "Connection must be established before setting up session"
        await conn.session.update(session=session_config)

        logger.info("Session configuration with function tools sent")

    async def _process_events(self):
        """Process events from the VoiceLive connection."""
        try:
            conn = self.connection
            assert conn is not None, "Connection must be established before processing events"
            async for event in conn:
                # print(f"DEBUG: Event received: {event.type}")
                await self._handle_event(event)

                # stop as soon as end_conversation fires
                if self._stop_event.is_set():
                    logger.info("Stop event set; breaking event loop.")
                    break
        except Exception:
            logger.exception("Error processing events")
            raise

    async def _handle_event(self, event):
        """Handle different types of events from VoiceLive."""
        print("Received event: ", event.type, " ",type(event.type))
        ap = self.audio_processor
        conn = self.connection
        assert ap is not None, "AudioProcessor must be initialized"
        assert conn is not None, "Connection must be established"

        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info("Session ready: %s", event.session.id)
            self.session_ready = True

            # Proactive greeting
            if not self.conversation_started:
                self.conversation_started = True
                logger.info("Sending proactive greeting request")
                try:
                    await conn.response.create()

                except Exception:
                    logger.exception("Failed to send proactive greeting request")

            # Start audio capture once session is ready
            ap.start_capture()

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            logger.info("User started speaking - stopping playback")
            print("🎤 Listening...")

            ap.skip_pending_audio()

            # Only cancel if response is active and not already done
            if self._active_response and not self._response_api_done:
                try:
                    await conn.response.cancel()
                    logger.debug("Cancelled in-progress response due to barge-in")
                except Exception as e:
                    if "no active response" in str(e).lower():
                        logger.debug("Cancel ignored - response already completed")
                    else:
                        logger.warning("Cancel failed: %s", e)

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            logger.info("🎤 User stopped speaking")
            print("🤔 Processing...")

        elif event.type == ServerEventType.RESPONSE_CREATED:
            logger.info("🤖 Assistant response created")
            self._active_response = True
            self._response_api_done = False

        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            logger.debug("Received audio delta")
            ap.queue_audio(event.delta)

        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            logger.info("🤖 Assistant finished speaking")
            print("🎤 Ready for next input...")

        elif event.type == "response.done":
            print("✅ Response complete")
            self._active_response = False
            self._response_api_done = True

            # Execute pending function call if arguments are ready
            # if self._pending_function_call and "arguments" in self._pending_function_call:
            await self._execute_function_call(self._pending_function_call)
            self._pending_function_call = None

        elif event.type == ServerEventType.ERROR:
            msg = event.error.message
            if "Cancellation failed: no active response" in msg:
                logger.debug("Benign cancellation error: %s", msg)
            else:
                logger.error("❌ VoiceLive error: %s", msg)
                print(f"Error: {msg}")

        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug("Conversation item created: %s", event.item.id)

            if event.item.type == ItemType.FUNCTION_CALL:
                function_call_item = event.item
                self._pending_function_call = {
                    "name": function_call_item.name,
                    "call_id": function_call_item.call_id,
                    "previous_item_id": function_call_item.id
                }
                print(f"🔧 Calling function: {function_call_item.name}")
                logger.info(f"Function call detected: {function_call_item.name} with call_id: {function_call_item.call_id}")

        elif event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            if self._pending_function_call and event.call_id == self._pending_function_call["call_id"]:
                logger.info(f"Function arguments received: {event.arguments}")
                self._pending_function_call["arguments"] = event.arguments

    async def _execute_function_call(self, function_call_info):
        """Execute a function call and send the result back to the conversation."""
        conn = self.connection
        assert conn is not None, "Connection must be established"
        
        if function_call_info is None:
            return
        function_name = function_call_info["name"]
        call_id = function_call_info["call_id"]
        previous_item_id = function_call_info["previous_item_id"]
        arguments = function_call_info["arguments"]

        try:
            if function_name in self.available_functions:
                logger.info(f"Executing function: {function_name}")
                result = self.available_functions[function_name](arguments)

                function_output = FunctionCallOutputItem(call_id=call_id, output=json.dumps(result))

                # Send result back to conversation
                await conn.conversation.item.create(previous_item_id=previous_item_id, item=function_output)
                logger.info(f"Function result sent: {result}")
                print(f"✅ Function {function_name} completed")

                # If we are ending, do NOT create another response
                if function_name == "end_conversation":
                    logger.info("Conversation ended by tool. Not creating another response.")
                    return

                await conn.response.create()

            else:
                logger.error(f"Unknown function: {function_name}")

        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")

    def upsert_person_profile(self, arguments=None) -> Dict[str, Any]:
        """Store structured personal profile information."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {"ok": False, "error": "INVALID_ARGUMENTS"}

        payload = arguments or {}
        # Minimal validation example
        person = payload.get("person", {})
        if not isinstance(person, dict):
            return {"ok": False, "error": "INVALID_PERSON_OBJECT"}

        # Merge patch semantics
        for k, v in person.items():
            if v is not None and v != "":
                self.person_profile[k] = v

        # TODO: persist to DB / encrypted store here
        logging.info(f"Stored person profile: {self.person_profile}")
        return {"ok": True, "stored_fields": list(person.keys())}
    
    def end_conversation(self, arguments=None) -> Dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        reason = (arguments or {}).get("reason", "unspecified")
        logging.info("end_conversation requested. reason=%s", reason)

        # Signal the event loop to stop
        try:
            self._stop_event.set()
        except Exception:
            logging.exception("Failed setting stop event")

        return {"ok": True, "ended": True, "reason": reason}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Assistant with Function Calling using Azure VoiceLive SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        help="Azure VoiceLive API key. If not provided, will use AZURE_VOICELIVE_API_KEY environment variable.",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_API_KEY"),
    )

    parser.add_argument(
        "--endpoint",
        help="Azure VoiceLive endpoint",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_ENDPOINT", "https://your-resource-name.services.ai.azure.com/"),
    )

    parser.add_argument(
        "--model",
        help="VoiceLive model to use",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_MODEL", "gpt-realtime"),
    )

    parser.add_argument(
        "--voice",
        help="Voice to use for the assistant. E.g. alloy, echo, fable, en-US-AvaNeural, en-US-GuyNeural",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-Ava:DragonHDLatestNeural"),
    )

    parser.add_argument(
        "--instructions",
        help="System instructions for the AI assistant",
        type=str,
        default=os.environ.get(
            "AZURE_VOICELIVE_INSTRUCTIONS", """You are a real-time voice intake agent on a live call with a client. Your job is to start the call, collect a small set of client details accurately, store them using the available tools, then end the session once complete.

            ROLE + TONE
            - Sound professional, warm, and efficient.
            - Keep the conversation short and focused.
            - Speak in clear, natural English.

            PRIMARY GOAL
            Collect and store the required client profile fields:
            1) person.full_name
            2) person.email
            3) person.phone
            4) person.company
            5) person.role
            6) person.location
            7) person.timezone

            (If the client does not have a company/role, store what they provide; do not force them.)

            WHAT TO STORE (ONLY THESE FIELDS)
            Store only these fields when the client provides them:
            - person.full_name
            - person.preferred_name
            - person.email
            - person.phone
            - person.timezone
            - person.location
            - person.company
            - person.role

            DATA COLLECTION RULES (MUST FOLLOW)
            1) Only store information the client explicitly states. Do NOT guess, infer, or fabricate.
            2) Use patch semantics: store only the fields just provided or corrected.
            3) If the client corrects something, update it immediately by storing the new value.
            4) Do NOT request or store highly sensitive attributes (health, religion, political affiliation, sexual life/orientation, etc.). If volunteered, do not store those details; acknowledge and move on.
            5) If the client refuses to answer a field, accept it and continue to the next field.

            TOOL USE (MANDATORY)
            - After each client answer that includes any profile data, call:
            upsert_person_profile(person={...}, source_utterance="short paraphrase")
            - Keep source_utterance short and non-sensitive (no long quotes).

            CALL OPENING (YOU MUST START THE CONVERSATION)
            Start immediately with a short introduction and purpose, then begin intake:
            “Hi, thanks for taking the call. I’m going to confirm a few details to make sure we have everything correct. Let’s start with your full name.”

            QUESTION FLOW (ASK ONLY WHAT’S MISSING)
            Ask for fields in this order, skipping anything already provided:
            1) Full name (and preferred name if they want)
            2) Company and role
            3) Email
            4) Phone
            5) Location (city/province/state/country as they prefer)
            6) Timezone (IANA like America/Toronto if they know it, otherwise “Eastern”, “Pacific”, etc.)

            CONFIRMATION
            After you have captured all required fields (or the client refused some), do a brief read-back summary:
            “Thanks. I have: [name], [company/role], [email], [phone], [location], [timezone]. Is that correct?”
            - If they correct anything, update via upsert_person_profile immediately, then confirm again briefly.

            COMPLETION + ENDING THE SESSION (MUST CLOSE)
            When either:
            A) all required fields are collected and stored, OR
            B) the client confirms the summary and there are no further corrections,
            then do the following:
            1) Say a short closing line:
            “Perfect — thank you. That’s everything I needed. Have a great day.”
            2) Immediately call the tool end_conversation with a clear reason, for example:
            end_conversation(reason="intake_complete")

            ERROR HANDLING
            - If a tool call fails, do not get stuck. Acknowledge briefly (“Thanks, one moment.”), proceed with the call, and continue collecting remaining fields.
            - If transcription is uncertain (e.g., email/phone), ask a quick verification:
            “Just to confirm, was that spelled … ?” / “Can you repeat the email/phone once more?”

            ABSOLUTE RULE
            Once intake is complete and confirmed, you MUST end the session by calling the tool end_conversation(reason="intake_complete").

            """
        ),
    )

    parser.add_argument(
        "--use-token-credential", help="Use Azure token credential instead of API key", action="store_true", default=False
    )

    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate credentials
    if not args.api_key and not args.use_token_credential:
        print("❌ Error: No authentication provided")
        print("Please provide an API key using --api-key or set AZURE_VOICELIVE_API_KEY environment variable,")
        print("or use --use-token-credential for Azure authentication.")
        sys.exit(1)

    # Create client with appropriate credential
    credential: Union[AzureKeyCredential, AsyncTokenCredential]
    if args.use_token_credential:
        credential = AzureCliCredential()
        logger.info("Using Azure token credential")
    else:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")

    # Create and start voice assistant with function calling
    client = AsyncFunctionCallingClient(
        endpoint=args.endpoint,
        credential=credential,
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
    )

    # Signal handlers for graceful shutdown
    def signal_handler(_sig, _frame):
        logger.info("Received shutdown signal")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        print("\n👋 Voice assistant shut down. Goodbye!")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for required dependencies
    dependencies = {
        "pyaudio": "Audio processing",
        "azure.ai.voicelive": "Azure VoiceLive SDK",
        "azure.core": "Azure Core libraries",
    }

    missing_deps = []
    for dep, description in dependencies.items():
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing_deps.append(f"{dep} ({description})")

    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install azure-ai-voicelive pyaudio python-dotenv")
        sys.exit(1)

    # Check audio system
    try:
        p = pyaudio.PyAudio()
        # Check for input devices
        input_devices = [
            i
            for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxInputChannels", 0) or 0) > 0
        ]
        # Check for output devices
        output_devices = [
            i
            for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxOutputChannels", 0) or 0) > 0
        ]
        p.terminate()

        if not input_devices:
            print("❌ No audio input devices found. Please check your microphone.")
            sys.exit(1)
        if not output_devices:
            print("❌ No audio output devices found. Please check your speakers.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Audio system check failed: {e}")
        sys.exit(1)

    print("🎙️  Voice Assistant with Function Calling - Azure VoiceLive SDK")
    print("=" * 65)

    # Run the assistant
    main()