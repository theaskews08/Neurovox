import asyncio
import logging
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperSTT:
    def __init__(self, resource_manager, model_size="medium"):
        self.resource_manager = resource_manager
        self.model_size = model_size
        self.model = None
        self.is_initialized = False
        self.mutex = asyncio.Lock()

    async def start(self):
        """Start the STT service"""
        logger.info(f"Starting Whisper STT with model size: {self.model_size}")

        # Allocate memory
        memory = await self.resource_manager.allocate("stt", 2048)
        logger.info(f"Allocated {memory}MB for STT")

        # Initialize the model
        try:
            # Load the model - use CUDA if available
            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="float16"
            )
            self.is_initialized = True
            logger.info("Whisper STT initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Whisper STT: {str(e)}")
            await self.resource_manager.release("stt")
            return False

    async def stop(self):
        """Stop the STT service"""
        logger.info("Stopping Whisper STT")
        self.is_initialized = False

        # Release memory
        await self.resource_manager.release("stt")

        # Free model (help garbage collection)
        self.model = None
        return True

    async def health_check(self):
        """Check if the service is healthy"""
        return self.is_initialized and self.model is not None

    async def listen(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        logger.info(f"Recording for {duration} seconds...")

        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        # Convert to the format expected by Whisper
        audio_data = recording.flatten()
        return audio_data

    async def transcribe(self, audio_data):
        """Transcribe audio data to text"""
        async with self.mutex:
            if not self.is_initialized:
                logger.error("Cannot transcribe: Whisper STT not initialized")
                return ""

            try:
                logger.info("Transcribing audio...")

                # Process with Whisper
                segments, info = self.model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                # Collect all segments into a single string
                text = " ".join(segment.text for segment in segments)

                logger.info(f"Transcription complete: {text}")
                return text

            except Exception as e:
                logger.error(f"Transcription error: {str(e)}")
                return ""