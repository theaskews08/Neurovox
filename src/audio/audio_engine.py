import os
import uuid
import asyncio
import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)


class AsyncAudioEngine:
    def __init__(self, temp_dir="temp_audio"):
        self.audio_queue = asyncio.Queue()
        self.is_playing = False
        self.current_player = None
        self.temp_dir = temp_dir
        self.temp_files = []
        self.mutex = asyncio.Lock()

        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)

    async def start(self):
        """Start the audio engine service"""
        logger.info("Starting audio engine...")
        return True

    async def stop(self):
        """Stop the audio engine service"""
        logger.info("Stopping audio engine...")
        await self.cleanup_all()
        return True

    async def play_audio(self, audio_data):
        """Queue audio data for playback"""
        await self.audio_queue.put(audio_data)

        # Start the player loop if not already running
        if not self.is_playing:
            asyncio.create_task(self._player_loop())

    async def _player_loop(self):
        """Background task to play queued audio"""
        self.is_playing = True

        while not self.audio_queue.empty():
            audio_data = await self.audio_queue.get()

            async with self.mutex:
                # Generate unique filename
                filename = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.wav")
                self.temp_files.append(filename)

                # Write audio data to file
                with open(filename, 'wb') as f:
                    f.write(audio_data)

                # Play the audio
                await self._play_file(filename)

                # Try to clean up the file
                await self._cleanup_file(filename)

        self.is_playing = False

    async def _play_file(self, filename):
        """Play audio file with proper resource management"""
        if platform.system() == "Windows":
            # Windows approach
            process = await asyncio.create_subprocess_exec(
                "powershell", "-c", f"(New-Object Media.SoundPlayer '{filename}').PlaySync()",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
        else:
            # Linux/Mac approach using aplay/afplay
            cmd = "afplay" if platform.system() == "Darwin" else "aplay"
            process = await asyncio.create_subprocess_exec(
                cmd, filename,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()

    async def _cleanup_file(self, filename):
        """Safely clean up temporary audio file"""
        for _ in range(3):  # Try up to 3 times
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    if filename in self.temp_files:
                        self.temp_files.remove(filename)
                    logger.debug(f"Successfully removed temporary file: {filename}")
                break
            except (PermissionError, OSError) as e:
                # File might still be in use, wait a bit
                logger.debug(f"Could not remove file {filename}, retrying... ({str(e)})")
                await asyncio.sleep(0.5)

    async def cleanup_all(self):
        """Clean up all temporary files"""
        async with self.mutex:
            for filename in list(self.temp_files):
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    self.temp_files.remove(filename)
                except (PermissionError, OSError) as e:
                    logger.warning(f"Could not remove temp file during cleanup: {filename} ({str(e)})")