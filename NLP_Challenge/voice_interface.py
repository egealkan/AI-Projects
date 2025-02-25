# from time import time
# import torch
# from transformers import pipeline
# from gtts import gTTS
# import sounddevice as sd
# import soundfile as sf
# import numpy as np
# import threading
# import queue
# import tempfile
# import os
# from typing import Optional, Generator
# import io

# class VoiceInterface:
#     def __init__(self):

#         if os.name == 'nt':  # For Windows
#             os.environ["PATH"] += os.pathsep + r'C:\Users\egmnl\OneDrive\Desktop\24-25 year\Fall Semester\Deep Learning\nlp challenge\ffmpeg-2024-12-04-git-2f95bc3cb3-full_build\bin'  # Adjust path if needed

#         # Initialize Whisper for STT
#         self.stt_pipeline = pipeline(
#             "automatic-speech-recognition",
#             model="openai/whisper-base",
#             device="cuda" if torch.cuda.is_available() else "cpu"
#         )
        
#         # Audio recording parameters
#         self.sample_rate = 16000
#         self.channels = 1
#         self.dtype = np.float32
        
#         # Voice activity detection parameters
#         self.silence_threshold = 0.01
#         self.silence_duration = 2.0  # seconds
#         self.min_speech_duration = 1.0  # seconds
#         self.max_speech_duration = 15.0  # seconds
        
#         # Recording state
#         self.is_recording = False
#         self.audio_queue = queue.Queue()

#     def _find_input_device(self):
#         """Find the first working microphone device."""
#         devices = sd.query_devices()
#         for device_idx, device in enumerate(devices):
#             if device['max_input_channels'] > 0:  # This is an input device
#                 try:
#                     # Test if we can open the device
#                     with sd.InputStream(device=device_idx, channels=1, samplerate=self.sample_rate):
#                         return device_idx
#                 except sd.PortAudioError:
#                     continue
#         return None

#     def record_audio(self) -> Optional[str]:
#         """Record audio with automatic endpoint detection."""
#         try:
#             # Find appropriate input device
#             device_id = self._find_input_device()
#             if device_id is None:
#                 print("No working microphone found!")
#                 return None

#             def audio_callback(indata, frames, time, status):
#                 if status:
#                     print(f"\nAudio callback error: {status}")
#                 self.audio_queue.put(indata.copy())

#             # Initialize recording buffers
#             audio_data = []
#             silence_frames = 0
#             speech_frames = 0
#             recording_start_time = time()
#             has_speech = False
            
#             with sd.InputStream(
#                 device=device_id,
#                 samplerate=self.sample_rate,
#                 channels=self.channels,
#                 dtype=self.dtype,
#                 callback=audio_callback
#             ) as stream:
#                 print("\nRecording started... Speak now.")
#                 self.is_recording = True
                
#                 while self.is_recording:
#                     try:
#                         audio_chunk = self.audio_queue.get(timeout=0.5).flatten()
#                         audio_data.extend(audio_chunk)
                        
#                         current_level = np.abs(audio_chunk).mean()
#                         if current_level > self.silence_threshold:
#                             speech_frames += len(audio_chunk)
#                             silence_frames = 0
#                             has_speech = True
#                         else:
#                             silence_frames += len(audio_chunk)
                        
#                         elapsed_time = time() - recording_start_time
#                         silence_time = silence_frames / self.sample_rate
                        
#                         if elapsed_time >= self.max_speech_duration or \
#                            (has_speech and silence_time >= self.silence_duration):
#                             break
                            
#                     except queue.Empty:
#                         continue

#             if not has_speech:
#                 return None

#             # Save and process audio
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
#                 audio_array = np.array(audio_data)
#                 sf.write(temp_file.name, audio_array, self.sample_rate)
#                 result = self.stt_pipeline(temp_file.name)
#                 transcribed_text = result["text"]
#                 os.unlink(temp_file.name)
#                 return transcribed_text.strip()

#         except Exception as e:
#             print(f"\nError during audio recording: {str(e)}")
#             return None
#         finally:
#             self.is_recording = False

#     def stop_recording(self):
#         """Stop the audio recording."""
#         self.is_recording = False

#     def text_to_speech_stream(self, text: str) -> Generator[bytes, None, None]:
#         """Convert text to speech using gTTS and stream the audio data."""
#         try:
#             # Split text into sentences for streaming
#             sentences = text.split('. ')
            
#             for sentence in sentences:
#                 if not sentence.strip():
#                     continue
                
#                 # Create gTTS object for the sentence
#                 tts = gTTS(text=sentence + '.', lang='en', slow=False)
                
#                 # Save to bytes buffer
#                 fp = io.BytesIO()
#                 tts.write_to_fp(fp)
#                 fp.seek(0)
                
#                 # Read audio data
#                 with sf.SoundFile(fp, 'r') as audio_file:
#                     while True:
#                         chunk = audio_file.read(8192)  # Read in chunks
#                         if not len(chunk):
#                             break
#                         yield (chunk * 32767).astype(np.int16).tobytes()
                
#         except Exception as e:
#             print(f"Error during text-to-speech conversion: {e}")
#             yield b''

#     def play_audio_stream(self, audio_stream: Generator[bytes, None, None]):
#         """Play streaming audio data."""
#         try:
#             with sd.OutputStream(
#                 samplerate=24000,  # gTTS default sample rate
#                 channels=1,
#                 dtype=np.int16
#             ) as stream:
#                 for audio_chunk in audio_stream:
#                     if audio_chunk:
#                         stream.write(np.frombuffer(audio_chunk, dtype=np.int16))
                        
#         except Exception as e:
#             print(f"Error during audio playback: {e}")
























from time import time
import torch
from transformers import pipeline
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import tempfile
import os
from typing import Optional, Generator
import io
import warnings
warnings.filterwarnings('ignore')

class VoiceInterface:
    def __init__(self):
        # For Windows ffmpeg path
        if os.name == 'nt':
            os.environ["PATH"] += os.pathsep + r'C:\Users\egmnl\OneDrive\Desktop\24-25 year\Fall Semester\Deep Learning\nlp challenge\ffmpeg-2024-12-04-git-2f95bc3cb3-full_build\bin'

        # Suppress specific warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        try:
            # Initialize Whisper for STT with minimal parameters
            self.stt_pipeline = pipeline(
                task="automatic-speech-recognition",
                model="openai/whisper-base",
                chunk_length_s=30,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            print(f"Error initializing Whisper model: {str(e)}")
            self.stt_pipeline = None
        
        # Audio recording parameters
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        
        # Voice activity detection parameters
        self.silence_threshold = 0.01
        self.silence_duration = 2.0  # seconds
        self.min_speech_duration = 1.0  # seconds
        self.max_speech_duration = 15.0  # seconds
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()

    def _find_input_device(self):
        """Find the first working microphone device."""
        devices = sd.query_devices()
        for device_idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # This is an input device
                try:
                    # Test if we can open the device
                    with sd.InputStream(device=device_idx, channels=1, samplerate=self.sample_rate):
                        return device_idx
                except sd.PortAudioError:
                    continue
        return None

    def record_audio(self) -> Optional[str]:
        """Record audio with automatic endpoint detection."""
        if self.stt_pipeline is None:
            print("Speech recognition model not initialized properly")
            return None

        try:
            # Find appropriate input device
            device_id = self._find_input_device()
            if device_id is None:
                print("No working microphone found!")
                return None

            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"\nAudio callback error: {status}")
                self.audio_queue.put(indata.copy())

            # Initialize recording buffers
            audio_data = []
            silence_frames = 0
            speech_frames = 0
            recording_start_time = time()
            has_speech = False
            
            with sd.InputStream(
                device=device_id,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=audio_callback
            ) as stream:
                print("\nRecording started... Speak now.")
                self.is_recording = True
                
                while self.is_recording:
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.5).flatten()
                        audio_data.extend(audio_chunk)
                        
                        current_level = np.abs(audio_chunk).mean()
                        if current_level > self.silence_threshold:
                            speech_frames += len(audio_chunk)
                            silence_frames = 0
                            has_speech = True
                            print(f"\nSpeech detected! Level: {current_level}")
                        else:
                            silence_frames += len(audio_chunk)
                        
                        elapsed_time = time() - recording_start_time
                        silence_time = silence_frames / self.sample_rate
                        
                        if elapsed_time >= self.max_speech_duration or \
                        (has_speech and silence_time >= self.silence_duration):
                            break
                            
                    except queue.Empty:
                        continue

            if not has_speech:
                print("\nNo speech detected.")
                return None

            # Convert audio data to numpy array
            audio_array = np.array(audio_data)

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                try:
                    # Write audio data to the temporary file
                    sf.write(temp_wav.name, audio_array, self.sample_rate)
                    
                    # Process audio using Whisper
                    result = self.stt_pipeline(temp_wav.name)
                    transcribed_text = result["text"]
                    return transcribed_text.strip()
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_wav.name)
                    except:
                        pass

        except Exception as e:
            print(f"\nError during audio recording: {str(e)}")
            return None
        finally:
            self.is_recording = False
            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

    # Rest of the class methods remain the same...
    def stop_recording(self):
        """Stop the audio recording."""
        self.is_recording = False

    def text_to_speech_stream(self, text: str) -> Generator[bytes, None, None]:
        """Convert text to speech using gTTS and stream the audio data."""
        try:
            # Split text into sentences for streaming
            sentences = text.split('. ')
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Create gTTS object for the sentence
                tts = gTTS(text=sentence + '.', lang='en', slow=False)
                
                # Save to bytes buffer
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                
                # Read audio data
                with sf.SoundFile(fp, 'r') as audio_file:
                    while True:
                        chunk = audio_file.read(8192)  # Read in chunks
                        if not len(chunk):
                            break
                        yield (chunk * 32767).astype(np.int16).tobytes()
                
        except Exception as e:
            print(f"Error during text-to-speech conversion: {e}")
            yield b''

    def play_audio_stream(self, audio_stream: Generator[bytes, None, None]):
        """Play streaming audio data."""
        try:
            with sd.OutputStream(
                samplerate=24000,  # gTTS default sample rate
                channels=1,
                dtype=np.int16
            ) as stream:
                for audio_chunk in audio_stream:
                    if audio_chunk:
                        stream.write(np.frombuffer(audio_chunk, dtype=np.int16))
                        
        except Exception as e:
            print(f"Error during audio playback: {e}")