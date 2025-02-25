from pathlib import Path
import sounddevice as sd
import numpy as np
import logging
import io
import wave
from google.cloud import speech
import os

# Build a path that starts from the current file's directory
MODEL_PATH = Path(__file__).parent / "models" / "gen-lang-client-0470544733-5d18af40aec7.json"

# Set the environment variable to that relative path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(MODEL_PATH)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging for RMS values

class DutchSTTService:
    def __init__(self, language_code="nl-BE", sample_rate=16000):
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.dtype = np.int16
        self.client = speech.SpeechClient()  # Make sure your credentials are set

    def record_audio(self, silence_threshold=1500, silence_duration=2.5):
        """
        Records audio continuously until the RMS amplitude stays below `silence_threshold`
        for `silence_duration` seconds. Returns a 1-D numpy array containing the audio samples.
        """
        logger.info("Recording audio with silence detection...")
        # Use a block size (number of samples per block)
        block_size = 4096
        # Calculate how many consecutive blocks constitute the silence duration.
        silence_blocks_needed = int((silence_duration * self.sample_rate) / block_size)
        silent_blocks = 0
        recorded = []

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=block_size,
            channels=1,
            dtype='int16'
        ) as stream:
            while True:
                data, overflow = stream.read(block_size)
                if overflow:
                    logger.warning("Audio buffer overflow!")
                recorded.append(data)
                # Convert raw bytes to a numpy array for RMS computation.
                np_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(np.square(np_data.astype(np.float32))))
                logger.debug(f"Block RMS: {rms}")  # Debug: log the RMS value

                # Check if the block is considered "silent"
                if rms < silence_threshold:
                    silent_blocks += 1
                else:
                    silent_blocks = 0

                # If we have enough consecutive silent blocks, stop recording.
                if silent_blocks >= silence_blocks_needed:
                    logger.info("Silence detected. Stopping recording.")
                    break

        # Combine all recorded blocks.
        recorded_bytes = b"".join(recorded)
        audio_array = np.frombuffer(recorded_bytes, dtype=np.int16)
        logger.info("Recording finished.")
        return audio_array

    def transcribe(self, silence_threshold=1500, silence_duration=2.5) -> str:
        """
        Records audio using silence detection and transcribes it using Google Cloud STT.
        Returns the transcription as a string.
        """
        # Record audio until silence is detected.
        audio_data = self.record_audio(silence_threshold=silence_threshold, silence_duration=silence_duration)

        # Create an in-memory WAV file from the recorded audio.
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 = 2 bytes per sample
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            wav_bytes = wav_buffer.getvalue()

        # Configure the audio for Google Cloud Speech.
        audio = speech.RecognitionAudio(content=wav_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
        )

        # Send the audio to Google Cloud STT (synchronous request).
        response = self.client.recognize(config=config, audio=audio)

        # Collect and return the transcription.
        transcription = ""
        for result in response.results:
            transcription += result.alternatives[0].transcript
        return transcription


