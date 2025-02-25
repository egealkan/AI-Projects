# import time
# import numpy as np
# import torch
# from PIL import Image, ImageGrab
# from pynput.keyboard import Controller, Key
# from fastai.vision.all import load_learner, PILImage
# from fastai.vision.all import Resize
# import os

# # Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')

# # Re-declare the custom functions (get_x, get_y, and classification_loss) used during training
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key']  # Only classify based on keyboard press

# def classification_loss(preds, targs):
#     return torch.nn.CrossEntropyLoss()(preds, targs.long())

# # Load the FastAI-trained model
# model_path = './models/model_classification_fastai.pkl'
# learn = load_learner(model_path)

# # Set up keyboard control
# keyboard = Controller()

# # Capture frames from the game and predict actions
# def capture_and_drive():
#     bbox = (0, 40, 1280, 720)  # Adjust this to match your game window size
#     while True:
#         # Capture the game screen
#         screen = np.array(ImageGrab.grab(bbox=bbox))

#         # Convert the NumPy array (frame) to a FastAI PILImage
#         frame = PILImage.create(screen)

#         # Apply the same preprocessing (resize, normalize, etc.)
#         frame = Resize(224)(frame)  # Apply resizing to 224x224 like in training

#         # Make prediction using FastAI's learner and only get the predicted label
#         predicted_action, _, _ = learn.predict(frame)  # Only get the predicted label (not the probabilities)

#         # # Print the predicted action (for debugging)
#         print(f"Predicted action: {predicted_action}")



#         # Perform the predicted action using special keys from pynput
#         if predicted_action == 'left':
#             keyboard.press(Key.left)
#             time.sleep(0.2)
#             keyboard.release(Key.left)
#         elif predicted_action == 'right':
#             keyboard.press(Key.right)
#             time.sleep(0.2)
#             keyboard.release(Key.right)
#         elif predicted_action == 'up':
#             keyboard.press(Key.up)
#             time.sleep(0.2)
#             keyboard.release(Key.up)
#         elif predicted_action == 'down':
#             keyboard.press(Key.down)
#             time.sleep(0.2)
#             keyboard.release(Key.down)
        
#         # Introduce a small delay to control the frame rate (adjust as necessary)
#         time.sleep(0.2)  # Increase sleep time to control FPS if necessary



# if __name__ == "__main__":
#     capture_and_drive()
























########################################################################################################################################

# from fastai.vision.all import *
# import torch
# from pynput.keyboard import Controller, Key
# from PIL import ImageGrab
# import numpy as np
# import cv2

# # Redefine custom functions used during training
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Load the trained model
# model_path = './models/model_classification_fastai.pkl'
# learn = load_learner(model_path)
# # print(learn.dls.vocab)

# # Initialize keyboard controller
# keyboard = Controller()

# # Function to capture frames and control the car
# def capture_and_drive():
#     bbox = (0, 40, 1280, 720)  # Adjust based on your screen setup

#     while True:
#         # Capture the game screen
#         screen = np.array(ImageGrab.grab(bbox=bbox))

#         # Convert to RGB and then to PIL image for FastAI
#         frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         frame = PILImage.create(frame)

#         # Resize to match training size
#         frame = Resize(224)(frame)

#         # Convert PIL image to tensor using ToTensor
#         frame_tensor = ToTensor()(frame)  # Convert to tensor

#         # Normalize the tensor to [0, 1] range (optional: depending on training settings)
#         frame_tensor = frame_tensor.float() / 255.0  # Convert to float and normalize if required

#         # Add batch dimension and move to device (GPU/CPU)
#         frame_tensor = frame_tensor.unsqueeze(0).to(learn.dls.device)

#         # Predict action using the loaded model
#         preds = learn.model(frame_tensor)
#         predicted_action = preds.argmax().item()

#         # Convert index to action
#         action = learn.dls.vocab[predicted_action]
#         print(f"Predicted action: {action}")

#         # action_vocab = ['down', 'left', 'right', 'up']
#         # action = action_vocab[predicted_action]
#         # print(f"Predicted action: {action}")

#         # Perform the predicted action
#         if action == 'left':
#             keyboard.press(Key.left)
#             time.sleep(0.1)
#             keyboard.release(Key.left)
#         elif action == 'right':
#             keyboard.press(Key.right)
#             time.sleep(0.1)
#             keyboard.release(Key.right)
#         elif action == 'up':
#             keyboard.press(Key.up)
#             time.sleep(0.1)
#             keyboard.release(Key.up)
#         elif action == 'down':
#             keyboard.press(Key.down)
#             time.sleep(0.1)
#             keyboard.release(Key.down)

#         # Control frame rate
#         time.sleep(0.2)

# if __name__ == "__main__":
#     capture_and_drive()

########################################################################################################################################
















########################################################################################################################################

from fastai.vision.all import *
import torch
from pynput.keyboard import Controller, Key
from PIL import ImageGrab
import numpy as np
import cv2

# Redefine custom functions used during training
def get_x(row):
    return row['frame_path']

def get_y(row):
    return row['key'].strip()

# Load the trained model
model_path = './models/model_classification_fastai_v2.pkl'
learn = load_learner(model_path)
# print(learn.dls.vocab)

# Initialize keyboard controller
keyboard = Controller()

# Function to capture frames and control the car
def capture_and_drive():
    bbox = (0, 40, 1280, 720)  # Adjust based on your screen setup

    while True:
        # Capture the game screen
        screen = np.array(ImageGrab.grab(bbox=bbox))

        # Convert to RGB and then to PIL image for FastAI
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = PILImage.create(frame)

        # Resize to match training size
        frame = Resize(224)(frame)

        # Convert PIL image to tensor using ToTensor
        frame_tensor = ToTensor()(frame)  # Convert to tensor

        # Normalize the tensor to [0, 1] range (optional: depending on training settings)
        frame_tensor = frame_tensor.float() / 255.0  # Convert to float and normalize if required

        # Add batch dimension and move to device (GPU/CPU)
        frame_tensor = frame_tensor.unsqueeze(0).to(learn.dls.device)

        # Predict action using the loaded model
        preds = learn.model(frame_tensor)
        predicted_action = preds.argmax().item()

        # Convert index to action
        action = learn.dls.vocab[predicted_action]
        print(f"Predicted action: {action}")

        # action_vocab = ['down', 'left', 'right', 'up']
        # action = action_vocab[predicted_action]
        # print(f"Predicted action: {action}")

        # Perform the predicted action
        if action == 'left':
            keyboard.press(Key.left)
            time.sleep(0.1)
            keyboard.release(Key.left)
        elif action == 'right':
            keyboard.press(Key.right)
            time.sleep(0.1)
            keyboard.release(Key.right)
        elif action == 'up':
            keyboard.press(Key.up)
            time.sleep(0.1)
            keyboard.release(Key.up)
        elif action == 'down':
            keyboard.press(Key.down)
            time.sleep(0.1)
            keyboard.release(Key.down)

        # Control frame rate
        time.sleep(0.2)

if __name__ == "__main__":
    capture_and_drive()
########################################################################################################################################
























































########################################################################################################################################



# from fastai.vision.all import *
# import torch
# from pynput.keyboard import Controller, Key
# from PIL import ImageGrab
# import numpy as np
# import cv2
# import time
# from collections import deque




# ##############################
# import warnings

# # Suppress all warnings
# warnings.filterwarnings("ignore")
# ##############################




# # Define the ResNetLSTM class again (must match the class used during training)
# class ResNetLSTM(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNetLSTM, self).__init__()
#         # Use a ResNet18 model with no pre-trained weights
#         self.resnet = resnet34(weights=None)  # Using weights=None to avoid loading pretrained weights
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Retain until last conv layer
        
#         # Adaptive pooling layer to reduce the spatial dimensions
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # This will output [batch_size, 512, 1, 1]
        
#         # LSTM to capture temporal information
#         self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True)
        
#         # Final fully connected layer
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         batch_size, seq_len, c, h, w = x.shape  # Input shape: [batch_size, seq_len, channels, height, width]
#         resnet_features = []
        
#         # Process each frame through ResNet
#         for t in range(seq_len):
#             with torch.no_grad():  # No gradient updates for ResNet
#                 feature_map = self.resnet(x[:, t])  # Shape: [batch_size, 512, 7, 7]
#                 pooled_map = self.pool(feature_map)  # Shape: [batch_size, 512, 1, 1]
#                 flattened_map = pooled_map.view(batch_size, 512)  # Shape: [batch_size, 512]
#                 resnet_features.append(flattened_map)
        
#         # Stack the features into a sequence
#         resnet_features = torch.stack(resnet_features, dim=1)  # Shape: [batch_size, seq_len, 512]
        
#         # Pass the sequence through LSTM
#         lstm_out, _ = self.lstm(resnet_features)  # Shape: [batch_size, seq_len, 128]
        
#         # Use the output from the last time step for classification
#         output = self.fc(lstm_out[:, -1, :])  # Shape: [batch_size, num_classes]
#         return output

# # Load the trained model (assuming you saved the entire model with torch.save)
# model_path = './models/full_model_resnet_lstm.pkl'
# model = torch.load(model_path)
# model.eval()  # Set the model to evaluation mode

# # Initialize keyboard controller
# keyboard = Controller()

# # Function to preprocess each frame (similar to training preprocessing)
# def preprocess_frame(frame):
#     """Resize and normalize the game screen frame as done during training."""
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = PILImage.create(frame)
#     img = img.resize((224, 224))  # Resize to 224x224
#     tensor_img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and permute to [C, H, W]
#     return tensor_img / 255.0  # Normalize the tensor to [0, 1]

# # Function to capture frames and control the car
# def capture_and_drive():
#     bbox = (0, 40, 1280, 720)  # Adjust based on your screen setup

#     # Use a deque (double-ended queue) to store the last 5 frames for the sequence
#     frame_sequence = deque(maxlen=5)  # We want to maintain a sliding window of 5 frames

#     while True:
#         # Capture the game screen
#         screen = np.array(ImageGrab.grab(bbox=bbox))

#         # Preprocess the captured frame (resize and normalize)
#         preprocessed_frame = preprocess_frame(screen)

#         # Add the preprocessed frame to the deque
#         frame_sequence.append(preprocessed_frame)

#         # If we have fewer than 5 frames, repeat the first frame to pad the sequence
#         if len(frame_sequence) < 5:
#             for _ in range(5 - len(frame_sequence)):
#                 frame_sequence.appendleft(frame_sequence[0])  # Repeat the first frame

#         # Stack the frames to create a sequence tensor [seq_len, C, H, W]
#         frame_sequence_tensor = torch.stack(list(frame_sequence), dim=0)

#         # Add batch dimension and move to device (CPU or GPU)
#         frame_sequence_tensor = frame_sequence_tensor.unsqueeze(0).to(next(model.parameters()).device)

#         # Predict action using the loaded model
#         with torch.no_grad():
#             preds = model(frame_sequence_tensor)
#         predicted_action = preds.argmax().item()

#         # Convert index to action (use your vocabulary or labels for action conversion)
#         action_vocab = ['down', 'left', 'right', 'up']  # Define your action vocabulary here
#         action = action_vocab[predicted_action]
#         print(f"Predicted action: {action}")

#         # Just for debugging
#         print(f"Predicted action index: {predicted_action}")
#         print(f"Action: {action_vocab[predicted_action]}")

#         # Perform the predicted action
#         if action == 'left':
#             keyboard.press(Key.left)
#             time.sleep(0.1)
#             keyboard.release(Key.left)
#         elif action == 'right':
#             keyboard.press(Key.right)
#             time.sleep(0.1)
#             keyboard.release(Key.right)
#         elif action == 'up':
#             keyboard.press(Key.up)
#             time.sleep(0.1)
#             keyboard.release(Key.up)
#         elif action == 'down':
#             keyboard.press(Key.down)
#             time.sleep(0.1)
#             keyboard.release(Key.down)

#         # Control frame rate
#         time.sleep(0.2)

# if __name__ == "__main__":
#     capture_and_drive()


########################################################################################################################################

















# from fastai.vision.all import *
# import torch
# from pynput.keyboard import Controller, Key
# from PIL import ImageGrab
# import numpy as np
# import cv2
# import time

# ##############################
# import warnings
# warnings.filterwarnings("ignore")
# ##############################

# # Define the SimpleResNet class again (must match the class used during training)
# class SimpleResNet(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleResNet, self).__init__()
#         # Use a ResNet18 model with no pre-trained weights
#         self.resnet = resnet18(weights=None)  # Using weights=None to avoid loading pretrained weights
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Retain until last conv layer
        
#         # Adaptive pooling layer to reduce the spatial dimensions
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # This will output [batch_size, 512, 1, 1]
        
#         # Final fully connected layer
#         self.fc = nn.Linear(512, 4)  # Adjusted for 4 actions: left, right, up, down

#     def forward(self, x):
#         # Process through ResNet
#         feature_map = self.resnet(x)
#         pooled_map = self.pool(feature_map)
#         flattened_map = pooled_map.view(x.size(0), -1)
#         output = self.fc(flattened_map)
#         return output

# # Load the trained model (assuming you saved the entire model with torch.save)
# model_path = './models/simple_resnet_model.pkl'
# model = torch.load(model_path)
# model.eval()  # Set the model to evaluation mode

# # Initialize keyboard controller
# keyboard = Controller()

# # Function to preprocess each frame (similar to training preprocessing)
# def preprocess_frame(frame):
#     """Resize and normalize the game screen frame as done during training."""
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = PILImage.create(frame)
#     img = img.resize((224, 224))  # Resize to 224x224
#     tensor_img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and permute to [C, H, W]
#     return tensor_img / 255.0  # Normalize the tensor to [0, 1]

# # Function to capture frames and control the car
# def capture_and_drive():
#     bbox = (0, 40, 1280, 720)  # Adjust based on your screen setup

#     while True:
#         # Capture the game screen
#         screen = np.array(ImageGrab.grab(bbox=bbox))

#         # Preprocess the captured frame (resize and normalize)
#         frame_tensor = preprocess_frame(screen).unsqueeze(0)

#         # Move to the model's device
#         frame_tensor = frame_tensor.to(next(model.parameters()).device)

#         # Predict action
#         with torch.no_grad():
#             preds = model(frame_tensor)
#         predicted_action = preds.argmax().item()

#         # Perform the predicted action
#         action_vocab = ['left', 'right', 'up', 'down']
#         action = action_vocab[predicted_action]
#         print(f"Predicted action: {action}")

#         # Perform the predicted action
#         if action == 'left':
#             keyboard.press(Key.left)
#             time.sleep(0.1)
#             keyboard.release(Key.left)
#         elif action == 'right':
#             keyboard.press(Key.right)
#             time.sleep(0.1)
#             keyboard.release(Key.right)
#         elif action == 'up':
#             keyboard.press(Key.up)
#             time.sleep(0.1)
#             keyboard.release(Key.up)
#         elif action == 'down':
#             keyboard.press(Key.down)
#             time.sleep(0.1)
#             keyboard.release(Key.down)

#         # Control frame rate
#         time.sleep(0.2)

# if __name__ == "__main__":
#     capture_and_drive()














# from fastai.vision.all import *
# import torch
# from pynput.keyboard import Controller, Key
# from PIL import ImageGrab
# import numpy as np
# import cv2
# import time

# # Redefine custom functions used during training
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Load the trained model
# model_path = './models/model_classification_fastai.pkl'
# learn = load_learner(model_path)

# # Initialize keyboard controller
# keyboard = Controller()

# # Function to preprocess each frame (similar to training preprocessing)
# def preprocess_frame(frame):
#     """Resize and normalize the game screen frame as done during training."""
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = PILImage.create(frame)  # Convert NumPy array to FastAI PILImage
#     img = img.resize((224, 224))  # Resize to 224x224
#     return img  # Return the preprocessed frame directly

# # Function to capture frames and control the car
# def capture_and_drive():
#     bbox = (0, 40, 1280, 720)  # Adjust based on your screen setup

#     while True:
#         # Always press "up" to accelerate the car
#         keyboard.press(Key.up)
#         time.sleep(0.1)  # Adjust this to control the speed of the car
#         keyboard.release(Key.up)

#         # Capture the game screen
#         screen = np.array(ImageGrab.grab(bbox=bbox))

#         # Preprocess the captured frame (resize and normalize)
#         preprocessed_frame = preprocess_frame(screen)

#         # Predict action using the loaded model
#         with torch.no_grad():
#             preds = learn.predict(preprocessed_frame)
#         predicted_action = preds[0]

#         # Convert index to action (based on your model's vocab)
#         action = str(predicted_action)  # Get the label directly from prediction
#         print(f"Predicted action: {action}")

#         # Perform the predicted action
#         if action == 'left':
#             keyboard.press(Key.left)
#             time.sleep(0.1)
#             keyboard.release(Key.left)
#         elif action == 'right':
#             keyboard.press(Key.right)
#             time.sleep(0.1)
#             keyboard.release(Key.right)

#         # Control frame rate
#         time.sleep(0.2)

# if __name__ == "__main__":
#     capture_and_drive()
