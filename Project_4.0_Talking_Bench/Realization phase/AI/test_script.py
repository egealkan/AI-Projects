import asyncio
import logging
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from modules.conversation_manager import ConversationManager
from services.tts_service import DutchTTSService
from services.stt_service import DutchSTTService  # Now using the new Google STT service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceChatTester:
    """Tester class focusing on voice interaction using Google Cloud STT and TTS."""
    
    async def run_test(self):
        try:
            # Initialize services
            conversation_manager = await ConversationManager.create()
            stt_service = DutchSTTService()  # New STT service using Google Cloud Speech
            tts_service = DutchTTSService()
            
            print("\n=== Starting Voice Chat Test ===")
            print("Press Ctrl+C to end the test\n")

            # Play welcome message
            welcome_message = conversation_manager.get_welcome_message()
            print(f"Bench: {welcome_message}")
            response = tts_service.synthesize_speech(welcome_message)
            if response["success"]:
                tts_service.play_audio()

            # Main conversation loop
            while True:
                print("\nListening... (speak in Dutch, or say 'stop' to end)")
                # Record and transcribe audio (recording)
                user_input = stt_service.transcribe(silence_threshold=1500, silence_duration=2.5)
                print(f"\nYou said: {user_input}")

                # Check for conversation ending keywords
                if user_input.lower() in ["stop", "doei", "tot ziens"]:
                    farewell = conversation_manager.get_farewell_message()
                    print(f"\nBench: {farewell}")
                    tts_service.synthesize_speech(farewell)
                    tts_service.play_audio()
                    break

                # Process conversation
                result = await conversation_manager.process_conversation(user_input)
                response_text = result.get("response", "")
                print(f"\nBench: {response_text}")
                response = tts_service.synthesize_speech(response_text)
                if response["success"]:
                    tts_service.play_audio()

                if result.get("metadata", {}).get("conversation_ended", False):
                    break

        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            logger.error(f"Test error: {e}")
            print(f"\nAn error occurred: {str(e)}")

async def main():
    tester = VoiceChatTester()
    await tester.run_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
