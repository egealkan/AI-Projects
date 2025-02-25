import numpy as np
import cv2
import pyautogui
import keyboard
import os
from datetime import datetime
import time

class GameScreenCapture:
    def __init__(self, output_dir='screenshots', interval=0.001):
        """
        Initialize screen capture with specified output directory and capture interval
        """
        self.output_dir = output_dir
        self.interval = interval  # Capture interval in seconds
        self.running = False
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created '{output_dir}' directory")
        
        # Optimize PyAutoGUI for faster screenshots
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
    
    def capture_screenshot(self):
        """
        Capture a screenshot and save it with timestamp
        """
        try:
            # Capture the screen using pyautogui
            screenshot = pyautogui.screenshot()
            
            # Convert to numpy array for OpenCV
            frame = np.array(screenshot)
            # Convert RGB to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Generate filename with timestamp and microseconds
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"trackmania_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            return True
        except Exception as e:
            print(f"Error capturing screenshot: {str(e)}")
            return False
    
    def start_capture(self):
        """
        Start continuous screen capture
        """
        print("Continuous screen capture started!")
        print("Press 'esc' to stop capturing.")
        
        self.running = True
        frames_captured = 0
        start_time = time.time()
        
        while self.running:
            # Capture screenshot
            if self.capture_screenshot():
                frames_captured += 1
            
            # Check for exit key
            if keyboard.is_pressed('esc'):
                self.running = False
            
            # Wait for the specified interval
            time.sleep(self.interval)
            
            # Print stats every 5 seconds
            if frames_captured % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frames_captured / elapsed_time
                print(f"Capturing at {fps:.2f} FPS")
        
        # Print final stats
        elapsed_time = time.time() - start_time
        fps = frames_captured / elapsed_time
        print(f"\nCapture stopped.")
        print(f"Total frames captured: {frames_captured}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Total time: {elapsed_time:.2f} seconds")

def main():
    # Initialize and start screen capture with 1ms interval
    # Note: Actual capture rate may be lower due to system limitations
    capture = GameScreenCapture(interval=0.001)
    capture.start_capture()

if __name__ == "__main__":
    main()