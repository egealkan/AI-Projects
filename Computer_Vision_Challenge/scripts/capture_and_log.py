# import cv2
# import numpy as np
# import os
# import csv
# import time
# import random  # Simulating speed data, replace this with actual speed retrieval method
# from pynput import keyboard
# from PIL import ImageGrab

# # Paths to save data
# FRAME_PATH = "./data/frames/"
# LOG_PATH = "./data/dataset.csv"

# # Ensure the frames folder exists
# if not os.path.exists(FRAME_PATH):
#     os.makedirs(FRAME_PATH)

# # Create CSV file for logging frames, key presses, and speed
# if not os.path.exists(LOG_PATH):
#     with open(LOG_PATH, mode='w') as file:
#         writer = csv.writer(file)
#         writer.writerow(["timestamp", "frame_path", "key", "speed"])  # Added speed to the header

# # Global variable to store the latest key press
# current_key = None

# def on_press(key):
#     """Callback to handle key press events."""
#     global current_key
#     try:
#         # Check if the key is alphanumeric (e.g., WASD for custom control)
#         current_key = key.char
#     except AttributeError:
#         # Handle special keys like 'left', 'right', 'up', etc.
#         if key == keyboard.Key.left:
#             current_key = 'left'
#         elif key == keyboard.Key.right:
#             current_key = 'right'
#         elif key == keyboard.Key.up:
#             current_key = 'up'
#         elif key == keyboard.Key.down:
#             current_key = 'down'
#         else:
#             current_key = None  # Ignore other keys
    
#     # Print to debug what keys are being registered
#     if current_key:
#         print(f"Key Pressed: {current_key}")

# def capture_and_log():
#     """Capture game frames, log the key presses, and capture speed."""
#     bbox = (0, 40, 1264, 720)  # Adjust this to match your game window size

#     while True:
#         # Capture the specified portion of the screen
#         screen = np.array(ImageGrab.grab(bbox=bbox))
#         frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

#         # Simulate retrieving speed data
#         speed = random.uniform(0, 200)  # Replace with actual speed retrieval method

#         # Save frame with unique timestamp
#         timestamp = str(int(time.time() * 1000))  # Milliseconds timestamp
#         frame_path = FRAME_PATH + f"frame_{timestamp}.jpg"
#         cv2.imwrite(frame_path, frame)

#         # Log the frame, key press, and speed (only if current_key is not None)
#         if current_key:
#             with open(LOG_PATH, mode='a') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([timestamp, frame_path, current_key, speed])  # Log key press and speed
        
#         # Introduce a small delay to control frame rate
#         time.sleep(0.1)  # 10 frames per second

# if __name__ == "__main__":
#     # Start listening to keyboard events
#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()

#     # Start capturing frames and logging data
#     capture_and_log()













########################################################################################################################################
# capture all the keys in the same folder

# import cv2
# import numpy as np
# from PIL import ImageGrab
# import os
# import time
# import csv
# import random  # Simulating speed data, replace with actual speed retrieval method
# from pynput import keyboard

# # Paths to save data
# FRAME_PATH = "./data/frames/"
# LOG_PATH = "./data/dataset.csv"

# # Ensure the frames folder exists
# if not os.path.exists(FRAME_PATH):
#     os.makedirs(FRAME_PATH)

# # Create CSV file for logging frames, key presses, and speed
# if not os.path.exists(LOG_PATH):
#     with open(LOG_PATH, mode='w') as file:
#         writer = csv.writer(file)
#         writer.writerow(["timestamp", "frame_path", "key", "speed"])  # Added speed to the header

# # Global variable to store the latest key press
# current_key = None

# def on_press(key):
#     """Callback to handle key press events."""
#     global current_key
#     try:
#         current_key = key.char  # Alphanumeric keys
#     except AttributeError:
#         # Special keys
#         if key == keyboard.Key.left:
#             current_key = 'left'
#         elif key == keyboard.Key.right:
#             current_key = 'right'
#         elif key == keyboard.Key.up:
#             current_key = 'up'
#         elif key == keyboard.Key.down:
#             current_key = 'down'
#         else:
#             current_key = None  # Ignore other keys

#     if current_key:
#         print(f"Key Pressed: {current_key}")

# def capture_and_log():
#     """Capture game frames, log the key presses, and capture speed."""
#     bbox = (0, 40, 1264, 720)  # Adjust to match your game window size

#     while True:
#         # Capture the screen
#         screen = np.array(ImageGrab.grab(bbox=bbox))
#         frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

#         # Crop to focus only on the road section (tweak these coordinates based on the actual region of the road)
#         cropped_frame = frame[300:450, 150:1150]  # Example y,x coordinates

#         # Simulate speed data
#         speed = random.uniform(0, 200)  # Replace with actual speed retrieval method

#         # Save frame with unique timestamp
#         timestamp = str(int(time.time() * 1000))  # Milliseconds timestamp
#         frame_path = FRAME_PATH + f"frame_{timestamp}.jpg"
#         cv2.imwrite(frame_path, cropped_frame)

#         # Log the frame, key press, and speed (only if current_key is not None)
#         if current_key:
#             with open(LOG_PATH, mode='a') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([timestamp, frame_path, current_key, speed])  # Log key press and speed

#         # Control frame rate
#         time.sleep(0.1)  # 10 frames per second

# if __name__ == "__main__":
#     # Start listening to keyboard events
#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()

#     # Start capturing frames and logging data
#     capture_and_log()
########################################################################################################################################













########################################################################################################################################
# capture all the keys in the different folder

import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
import csv
import random  # Simulating speed data, replace with actual speed retrieval method
from pynput import keyboard

# Paths to save data
BASE_FRAME_PATH = "./data/frames/"
LOG_PATH = "./data/dataset.csv"

# Ensure the frame directories exist for each key
for key in ['up', 'down', 'left', 'right']:
    key_dir = os.path.join(BASE_FRAME_PATH, key)
    if not os.path.exists(key_dir):
        os.makedirs(key_dir)

# Create CSV file for logging frames, key presses, and speed
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "frame_path", "key", "speed"])  # Added speed to the header

# Global variable to store the latest key press
current_key = None

def on_press(key):
    """Callback to handle key press events."""
    global current_key
    try:
        current_key = key.char  # Alphanumeric keys
    except AttributeError:
        # Special keys
        if key == keyboard.Key.left:
            current_key = 'left'
        elif key == keyboard.Key.right:
            current_key = 'right'
        elif key == keyboard.Key.up:
            current_key = 'up'
        elif key == keyboard.Key.down:
            current_key = 'down'
        else:
            current_key = None  # Ignore other keys

    if current_key:
        print(f"Key Pressed: {current_key}")

def capture_and_log():
    """Capture game frames, log the key presses, and capture speed."""
    bbox = (0, 40, 1264, 720)  # Adjust to match your game window size

    while True:
        # Capture the screen
        screen = np.array(ImageGrab.grab(bbox=bbox))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        # Crop to focus only on the road section (tweak these coordinates based on the actual region of the road)
        cropped_frame = frame[300:450, 150:1150]  # Example y,x coordinates

        # Simulate speed data
        speed = random.uniform(0, 200)  # Replace with actual speed retrieval method

        # Save frame with unique timestamp
        timestamp = str(int(time.time() * 1000))  # Milliseconds timestamp
        
        # Check if the current key is one of the valid keys
        if current_key in ['up', 'down', 'left', 'right']:
            # Save the frame to the corresponding directory
            frame_dir = os.path.join(BASE_FRAME_PATH, current_key)
            frame_path = os.path.join(frame_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(frame_path, cropped_frame)

            # Log the frame, key press, and speed
            with open(LOG_PATH, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, frame_path, current_key, speed])  # Log key press and speed

        # Control frame rate
        time.sleep(0.1)  # 10 frames per second

if __name__ == "__main__":
    # Start listening to keyboard events
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start capturing frames and logging data
    capture_and_log()
########################################################################################################################################