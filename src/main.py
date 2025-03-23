import asyncio
import logging
from pathlib import Path
import os

# Local imports
from audio.audio_engine import AsyncAudioEngine
from audio.speech_to_text import WhisperSTT
from audio.text_to_speech import KokoroTTS
from brain.conversation_manager import ConversationalBrain
from brain.memory_manager import MemoryManager
from resources.resource_manager import NeuralResourceManager
from services.service_manager import ServiceManager
from utils.logging_utils import setup_logging

# Setup logging
logger = setup_logging()


class NeuroVox:
    def __init__(self):
        # Create directories if they don't exist
        self.data_dir = Path("../data")
        self.temp_audio_dir = self.data_dir / "temp_audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the resource manager
        self.resource_manager = NeuralResourceManager(gpu_memory=8192)  # 8GB in MB

        # Initialize components
        self.audio_engine = AsyncAudioEngine(temp_dir=str(self.temp_audio_dir))
        self.stt = WhisperSTT(resource_manager=self.resource_manager)
        self.tts = KokoroTTS(resource_manager=self.resource_manager)
        self.memory_manager = MemoryManager()
        self.brain = ConversationalBrain(resource_manager=self.resource_manager,
                                         memory_manager=self.memory_manager)

        # Initialize service manager
        self.service_manager = ServiceManager()
        self._register_services()

    def _register_services(self):
        """Register all services with the service manager"""
        self.service_manager.register_service(
            "stt", self.stt.start, self.stt.stop, self.stt.health_check)
        self.service_manager.register_service(
            "tts", self.tts.start, self.tts.stop, self.tts.health_check)
        self.service_manager.register_service(
            "brain", self.brain.start, self.brain.stop, self.brain.health_check,
            dependencies=["stt", "tts"])
        self.service_manager.register_service(
            "audio", self.audio_engine.start, self.audio_engine.stop)

    async def start(self):
        """Start the voice assistant"""
        logger.info("Starting NeuroVox voice assistant...")
        await self.service_manager.start_all()
        logger.info("NeuroVox started successfully!")

        # Main interaction loop
        try:
            while True:
                # Record audio
                logger.info("Listening...")
                audio_data = await self.stt.listen()

                # Process speech to text
                text = await self.stt.transcribe(audio_data)
                logger.info(f"User said: {text}")

                if not text.strip():
                    logger.info("No speech detected, continuing...")
                    continue

                # Process with brain
                response = await self.brain.process_utterance(text)
                logger.info(f"Assistant response: {response}")

                # Generate speech
                audio_response = await self.tts.synthesize(response)

                # Play response
                await self.audio_engine.play_audio(audio_response)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.service_manager.stop_all()
            logger.info("NeuroVox stopped.")


async def main():
    assistant = NeuroVox()
    await assistant.start()


if __name__ == "__main__":
    asyncio.run(main())