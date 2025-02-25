import pygame
import logging
import threading
import time
from google.cloud import texttospeech
import os
from pathlib import Path

# Build a path that starts from the current file's directory
MODEL_PATH = Path(__file__).parent / "models" / "gen-lang-client-0470544733-5d18af40aec7.json"

# Set the environment variable to that relative path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(MODEL_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DutchTTSService:
    def __init__(self):
        self._initialize_pygame()
        self._initialize_google_client()
        
        # Audio file path
        self.audio_file = "temp_audio.mp3"
        self._lock = threading.Lock()
        
        logger.info("Dutch TTS Service initialized")

    def _initialize_pygame(self):
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=24000)
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            raise

    def _initialize_google_client(self):
        try:
            self.client = texttospeech.TextToSpeechClient()
            self.voice_config = texttospeech.VoiceSelectionParams(
                language_code="nl-BE",
                name="nl-BE-Standard-D",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            self.audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
                volume_gain_db=1.0
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud TTS client: {e}")
            raise

    def synthesize_speech(self, text: str) -> dict:
        try:
            with self._lock:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                response = self.client.synthesize_speech(
                    input=synthesis_input,
                    voice=self.voice_config,
                    audio_config=self.audio_config
                )

                with open(self.audio_file, "wb") as out:
                    out.write(response.audio_content)

                logger.info(f"Speech synthesized: '{text[:50]}...'")
                return {"success": True}
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return {"success": False, "error": str(e)}

    def play_audio(self):
        try:
            with self._lock:
                if not os.path.exists(self.audio_file):
                    logger.error("Audio file not found")
                    return

                pygame.mixer.music.load(self.audio_file)
                pygame.mixer.music.play()

            # Wait until playback is finished
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
        finally:
            self._cleanup_playback()

    def _cleanup_playback(self):
        with self._lock:
            pygame.mixer.music.unload()

    def cleanup(self):
        try:
            if os.path.exists(self.audio_file):
                os.remove(self.audio_file)
            pygame.mixer.quit()
            logger.info("TTS Service cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        self.cleanup()
