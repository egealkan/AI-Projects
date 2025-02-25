# from fastai.vision.all import *
# import torch
# from torch.nn import CrossEntropyLoss
# import pandas as pd
# import os
# from fastai.metrics import accuracy

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Fix the frame paths in the dataset (ensures that frame paths are correct)
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Define your `get_x` (image path) and `get_y` (classification labels: keyboard presses only)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key']  # Only classify based on keyboard press

# # Create a DataBlock for classification task only
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),  # Input is an image, output is classification
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),  # Split 20% of data for validation
#     item_tfms=Resize(224),  # Resize image to 224x224
#     batch_tfms=aug_transforms(mult=2)  # Optional data augmentations
# )

# # Load the DataFrame into FastAI's DataLoader with no workers to ensure no multiprocessing
# dls = block.dataloaders(data, bs=16, num_workers=0)

# # Determine number of unique keys (classification task)
# num_classes = dls.c  # This gives the number of classes from the DataLoader

# # Custom classification loss function
# def classification_loss(preds, targs):
#     return CrossEntropyLoss()(preds, targs.long())

# # Define the model, specifying `n_out` for classification
# model = vision_learner(dls, resnet34, n_out=num_classes, loss_func=classification_loss, metrics=[accuracy])

# # Train the model
# model.fine_tune(5)

# # Move the model to CPU before saving
# model.to('cpu')

# # Save the trained model
# model.save('model_classification_fastai.pkl')

# print("Training complete and model saved.")











# from fastai.vision.all import *
# import torch
# from torch.nn import CrossEntropyLoss
# import pandas as pd
# import os
# from fastai.metrics import accuracy

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Fix the frame paths in the dataset
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Define `get_x` (image path) and `get_y` (classification labels: keyboard presses only)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()  # Only classify based on keyboard press

# # Create a DataBlock for classification task only
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),  # Input is an image, output is classification
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),  # Split 20% of data for validation
#     item_tfms=Resize(224),  # Resize image to 224x224
#     batch_tfms=aug_transforms(mult=2)  # Optional data augmentations
# )

# # Load the DataFrame into FastAI's DataLoader with no workers to ensure no multiprocessing
# dls = block.dataloaders(data, bs=16, num_workers=0)

# # Use one_batch to fetch a single batch and examine the labels
# x, y = dls.one_batch()
# print(f"Labels in this batch: {y}")


# # Determine number of unique keys (classification task)
# num_classes = dls.c  # This gives the number of classes from the DataLoader

# # Custom classification loss function
# def classification_loss(preds, targs):
#     return CrossEntropyLoss()(preds, targs.long())

# # Define the model, specifying `n_out` for classification
# learn = vision_learner(dls, resnet34, n_out=num_classes, loss_func=classification_loss, metrics=[accuracy])

# # Train the model
# learn.fine_tune(10)

# # Move the model to CPU before saving
# learn.to('cpu')

# # Save the trained model using FastAI's `export` method (not just the weights but the entire model)
# learn.export('./models/model_classification_fastai.pkl')

# print("Training complete and model saved.")






















########################################################################################################################################

# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy, Precision, Recall, F1Score

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Ensure the paths to the images are correct
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Define `get_x` (image path) and `get_y` (classification labels: keyboard presses)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Create a DataBlock for classification task
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),  
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),  
#     item_tfms=Resize(224),  
#     batch_tfms=aug_transforms(mult=2, do_flip=False, flip_vert=False)
#     # batch_tfms=aug_transforms(mult=2, do_flip=True, max_rotate=30.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2)  
# )

# # Create the DataLoader
# dls = block.dataloaders(data, bs=16, num_workers=0)

# ################################################
# # Calculate class weights based on label distribution
# # Count occurrences of each class
# # label_counts = data['key'].value_counts()
# # total_samples = len(data)

# # Compute inverse frequency for each class
# # weights = torch.tensor([total_samples / label_counts[cls] for cls in dls.vocab], dtype=torch.float32)
# # weights = torch.tensor([1.5, 1.5, 1.0, 1.0], dtype=torch.float32)  # Slightly higher weights for left/right

# # Create CrossEntropyLoss with class weights
# # loss_func = CrossEntropyLoss(weight=weights)
# ################################################

# # Define and train the model
# learn = vision_learner(dls, resnet34, metrics=[accuracy])
# # learn = vision_learner(dls, resnet34, pretrained=False, metrics=[accuracy])
# # learn = vision_learner(dls, resnet18, pretrained=False, loss_func=loss_func, metrics=[accuracy])
# # learn = vision_learner(dls, resnet18, pretrained=False, loss_func=loss_func, metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')])


# # Fine-tune the model for epochs
# learn.fine_tune(50)

# # Save the model
# learn.export('./models/model_classification_fastai.pkl')


########################################################################################################################################




















########################################################################################################################################
# with advanced techniques

# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy, Precision, Recall, F1Score

# ##############################
# import warnings
# # Suppress all warnings
# warnings.filterwarnings("ignore")
# ##############################

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Ensure the paths to the images are correct
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Define `get_x` (image path) and `get_y` (classification labels: keyboard presses)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Create a DataBlock for classification task
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),  
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),  
#     item_tfms=Resize(224),  
#     batch_tfms=aug_transforms(mult=2, do_flip=False, flip_vert=False, max_rotate=30.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2)
# )

# # Create the DataLoader
# dls = block.dataloaders(data, bs=16, num_workers=0)

# # Define and train the model
# learn = vision_learner(dls, resnet34, pretrained=True, metrics=[accuracy])

# # Find the optimal learning rate
# learn.lr_find()

# # Unfreeze model for training the entire architecture
# learn.unfreeze()

# # Fine-tune the model with one-cycle learning and differential learning rates
# learn.fit_one_cycle(30, lr_max=slice(1e-6, 1e-4))

# # Apply mixed precision for faster training
# learn = learn.to_fp16()

# # Save the model
# learn.export('./models/model_classification_fastai_v2.pkl')




# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy, Precision, Recall, F1Score

# # ##############################
# import warnings
# # Suppress all warnings
# warnings.filterwarnings("ignore")
# # ##############################

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Ensure the paths to the images are correct
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Define `get_x` (image path) and `get_y` (classification labels: keyboard presses)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Create a DataBlock for classification task
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),
#     item_tfms=Resize(224),
#     batch_tfms=aug_transforms(mult=2, do_flip=False, flip_vert=False, max_rotate=30.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2)
# )

# # Create the DataLoader
# dls = block.dataloaders(data, bs=16, num_workers=0)

# # Define the learner
# learn = vision_learner(dls, resnet34, metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')]).to_fp16()

# # Step 1: Find the optimal learning rate
# learn.lr_find()

# # Step 2: Train with a selected learning rate
# # Choose a learning rate from the learning rate plot (e.g., 1e-3)
# learn.fine_tune(30, base_lr=1e-3)

# # Step 3: Unfreeze the model and apply differential learning rates
# # Fine-tune the entire model with different learning rates for earlier and later layers
# learn.unfreeze()
# learn.fit_one_cycle(10, slice(1e-5, 1e-3))

# # Step 4: Save the model
# learn.export('./models/model_classification_fastai_v2.pkl')

########################################################################################################################################









########################################################################################################################################
# with advanced techniques and over and under sampling

# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy, Precision, Recall, F1Score
# from sklearn.utils import resample
# import warnings
# import matplotlib.pyplot as plt

# ##############################
# # Suppress all warnings
# warnings.filterwarnings("ignore")
# ##############################

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Ensure the paths to the images are correct
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Oversample the "left" and "right" classes specifically to match the size of the largest class
# left_class = data[data['key'] == 'left']
# right_class = data[data['key'] == 'right']
# up_class = data[data['key'] == 'up']
# down_class = data[data['key'] == 'down']

# # Find the size of the largest class to match it during oversampling
# largest_class_size = max(len(up_class), len(down_class), len(left_class), len(right_class))

# # Perform oversampling for "left" and "right" classes
# left_upsampled = resample(left_class, replace=True, n_samples=largest_class_size, random_state=42)
# right_upsampled = resample(right_class, replace=True, n_samples=largest_class_size, random_state=42)

# # Combine the resampled and original classes
# data_balanced = pd.concat([up_class, down_class, left_upsampled, right_upsampled])

# # Shuffle the dataset to ensure randomness
# data_balanced = data_balanced.sample(frac=1).reset_index(drop=True)

# # Define `get_x` (image path) and `get_y` (classification labels: keyboard presses)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Create a DataBlock for classification task with the balanced dataset
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),  
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),  
#     # item_tfms=Resize(224),  
#     batch_tfms=[
#         RandomResizedCrop(224, min_scale=0.8),  # Randomly crop and resize images directly to 224x224
#         *aug_transforms(do_flip=False, flip_vert=False)
#     ]
# )

# # Create the DataLoader with the balanced data
# dls = block.dataloaders(data_balanced, bs=16, num_workers=0)

# # Define and train the model
# learn = vision_learner(dls, resnet34, pretrained=True, metrics=[accuracy], ps=0.2)  # `ps` adds dropout

# # Find the optimal learning rate
# learn.lr_find()

# # Unfreeze model for training the entire architecture
# learn.unfreeze()

# # Fine-tune the model with one-cycle learning and differential learning rates
# learn.fit_one_cycle(30, lr_max=slice(1e-5, 1e-3))

# # Apply mixed precision for faster training (optional)
# learn = learn.to_fp16()

# # Visualize results after training
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix(figsize=(8, 8), dpi=100)
# interp.plot_top_losses(9, figsize=(15, 10))
# plt.show()

# # Save the model
# learn.export('./models/model_classification_fastai_v2.pkl')
########################################################################################################################################




















########################################################################################################################################
# advanced techniques with individual folders for each key press

from fastai.vision.all import *
import os
from torch.nn import CrossEntropyLoss
from fastai.metrics import accuracy, Precision, Recall, F1Score
import warnings
import matplotlib.pyplot as plt

##############################
# Suppress all warnings
warnings.filterwarnings("ignore")
##############################

# Paths
BASE_FRAME_PATH = "./data/frames/"
LOG_PATH = "./data/dataset.csv"

# Define the function to extract data directly from folders
def get_image_data():
    """Loads image data from folders for each label."""
    # Create a list of all image paths and their corresponding labels based on folder names
    image_files = get_image_files(BASE_FRAME_PATH)
    data = pd.DataFrame({
        'frame_path': [str(f) for f in image_files],
        'key': [f.parent.name for f in image_files]  # Extract the folder name as the label
    })
    return data

# Load the data
data = get_image_data()

# Define `get_x` (image path) and `get_y` (classification labels: keyboard presses)
def get_x(row):
    return row['frame_path']

def get_y(row):
    return row['key'].strip()

# Create a DataBlock for classification task with data loaded from folders
block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  
    get_x=get_x,
    get_y=get_y,
    splitter=RandomSplitter(valid_pct=0.2),  
    batch_tfms=[
        RandomResizedCrop(224, min_scale=0.8),  # Randomly crop and resize images directly to 224x224
        *aug_transforms(do_flip=False, flip_vert=False)
    ]
)

# Create the DataLoader with the balanced data
dls = block.dataloaders(data, bs=16, num_workers=0)

# Define and train the model
learn = vision_learner(dls, resnet50, pretrained=True, metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')], ps=0.2)  # `ps` adds dropout

# Find the optimal learning rate
learn.lr_find()

# Unfreeze model for training the entire architecture
learn.unfreeze()

# Fine-tune the model with one-cycle learning and differential learning rates
learn.fit_one_cycle(50, lr_max=slice(1e-4, 1e-2))

# Apply mixed precision for faster training (optional)
learn = learn.to_fp16()

# Visualize results after training
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8, 8), dpi=100)
interp.plot_top_losses(9, figsize=(15, 10))
plt.show()

# Save the model
learn.export('./models/model_classification_fastai_v2.pkl')

########################################################################################################################################




























########################################################################################################################################
# with over and under sampling

# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch import nn
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy, Precision, Recall, F1Score
# from PIL import Image  # Ensure correct image handling
# import numpy as np
# import torch
# from sklearn.utils import resample  # For resampling the data

# ##############################
# import warnings
# # Suppress all warnings
# warnings.filterwarnings("ignore")
# ##############################

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Ensure the paths to the images are correct
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Resample and augment the "down" and "left" classes
# down_class = data[data['key'] == 'down']
# left_class = data[data['key'] == 'left']
# up_class = data[data['key'] == 'up']
# right_class = data[data['key'] == 'right']

# # We will oversample "down" and "left" to match the size of the largest class
# largest_class_size = max(len(up_class), len(right_class))

# # Augment the "down" and "left" classes using simple duplication and transformations
# down_upsampled = resample(down_class, replace=True, n_samples=largest_class_size, random_state=42)
# left_upsampled = resample(left_class, replace=True, n_samples=largest_class_size, random_state=42)

# # Combine the augmented data with the other classes
# data_balanced = pd.concat([up_class, right_class, down_upsampled, left_upsampled])

# # Shuffle the dataset to mix augmented data with the original data
# data_balanced = data_balanced.sample(frac=1).reset_index(drop=True)

# # Define get_x (image path) and get_y (classification labels: keyboard presses)
# def get_x(row):
#     # Get a sequence of 5 frames
#     frames = get_sequence(row.name, window=5)
#     # Convert each frame to a tensor manually using PyTorch from PIL
#     frame_tensors = torch.stack([convert_frame(f) for f in frames], dim=0)
#     return frame_tensors

# def get_y(row):
#     return row['key'].strip()

# # Custom function to retrieve sequences of frames
# def get_sequence(idx, window=5):
#     """Returns a sequence of frames (with a window) instead of a single frame."""
#     start_idx = max(0, idx - window + 1)
#     end_idx = idx + 1
#     frames = data_balanced.iloc[start_idx:end_idx]['frame_path'].values
#     # Ensure all frames have the same size by padding with the first frame if needed
#     if len(frames) < window:
#         padding = [frames[0]] * (window - len(frames))
#         frames = list(padding) + list(frames)
#     return frames

# # Helper function to resize and convert each image to a tensor
# def convert_frame(frame_path):
#     """Resize and convert the frame to a tensor compatible with the FastAI pipeline."""
#     img = Image.open(frame_path).convert('RGB')  # Ensure it's in RGB mode
#     img = img.resize((224, 224))  # Resize to match expected input size
#     tensor_img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and permute to [C, H, W]
#     return tensor_img / 255.0  # Normalize to [0, 1]

# # Custom collation function to handle batches of sequences
# def collate_fn(batch):
#     """Collates sequences of frames into a single tensor batch."""
#     xs, ys = zip(*batch)  # Unpack the batch into inputs and labels
#     # Stack the sequences into a batch of tensors
#     x_batch = torch.stack(xs)
#     y_batch = torch.tensor([dls.vocab.index(y) for y in ys])
#     return x_batch, y_batch

# # Custom DataBlock with Data Augmentation
# block = DataBlock(
#     blocks=(TransformBlock, CategoryBlock),
#     get_x=get_x,  # Returns stacked tensors of sequences
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),
#     item_tfms=Resize(224),
#     batch_tfms=aug_transforms(mult=2, do_flip=True, max_rotate=30.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2)
# )

# # Create the DataLoader with the custom collate function
# dls = block.dataloaders(data_balanced, bs=8, num_workers=0, collate_fn=collate_fn)

# # Define the LSTM-based model that uses ResNet for feature extraction and LSTM for sequential context
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

# # Define learner for training the ResNet-LSTM model
# class CustomLearner(Learner):
#     def __init__(self, dls, model, metrics):
#         super().__init__(dls, model, loss_func=CrossEntropyLoss(), metrics=metrics)

# # Create the model and learner
# num_classes = len(dls.vocab)
# model = ResNetLSTM(num_classes=num_classes)

# learn = CustomLearner(dls, model, metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')])

# # Find the optimal learning rate
# learn.lr_find()

# # Use the optimal learning rate to fine-tune the model
# learn.fit_one_cycle(30, lr_max=1e-3)  # Adjust `lr_max` based on lr_find results

# # Save the entire model using torch.save with .pkl extension
# torch.save(model, './models/full_model_resnet_lstm.pkl')

########################################################################################################################################













########################################################################################################################################
# with no over and under sampling

# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch import nn
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy, Precision, Recall, F1Score
# from PIL import Image  # Ensure correct image handling
# import numpy as np
# import torch

# ##############################
# import warnings
# # Suppress all warnings
# warnings.filterwarnings("ignore")
# ##############################

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Ensure the paths to the images are correct
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Define get_x (image path) and get_y (classification labels: keyboard presses)
# def get_x(row):
#     # Get a sequence of 5 frames
#     frames = get_sequence(row.name, window=5)
#     # Convert each frame to a tensor manually using PyTorch from PIL
#     frame_tensors = torch.stack([convert_frame(f) for f in frames], dim=0)
#     return frame_tensors

# def get_y(row):
#     return row['key'].strip()

# # Custom function to retrieve sequences of frames
# def get_sequence(idx, window=5):
#     """Returns a sequence of frames (with a window) instead of a single frame."""
#     start_idx = max(0, idx - window + 1)
#     end_idx = idx + 1
#     frames = data.iloc[start_idx:end_idx]['frame_path'].values
#     # Ensure all frames have the same size by padding with the first frame if needed
#     if len(frames) < window:
#         padding = [frames[0]] * (window - len(frames))
#         frames = list(padding) + list(frames)
#     return frames

# # Helper function to resize and convert each image to a tensor
# def convert_frame(frame_path):
#     """Resize and convert the frame to a tensor compatible with the FastAI pipeline."""
#     img = Image.open(frame_path).convert('RGB')  # Ensure it's in RGB mode
#     img = img.resize((224, 224))  # Resize to match expected input size
#     tensor_img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and permute to [C, H, W]
#     return tensor_img / 255.0  # Normalize to [0, 1]

# # Custom collation function to handle batches of sequences
# def collate_fn(batch):
#     """Collates sequences of frames into a single tensor batch."""
#     xs, ys = zip(*batch)  # Unpack the batch into inputs and labels
#     # Stack the sequences into a batch of tensors
#     x_batch = torch.stack(xs)
#     y_batch = torch.tensor([dls.vocab.index(y) for y in ys])
#     return x_batch, y_batch

# # Custom DataBlock without flip augmentation
# block = DataBlock(
#     blocks=(TransformBlock, CategoryBlock),
#     get_x=get_x,  # Returns stacked tensors of sequences
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),
#     item_tfms=Resize(224),  # Resize to 224x224
#     batch_tfms=aug_transforms(mult=2, do_flip=False, flip_vert=False)  # Disable flips
# )

# # Create the DataLoader with the custom collate function
# dls = block.dataloaders(data, bs=8, num_workers=0, collate_fn=collate_fn)

# # Define the LSTM-based model that uses ResNet for feature extraction and LSTM for sequential context
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

# # Define learner for training the ResNet-LSTM model
# class CustomLearner(Learner):
#     def __init__(self, dls, model, metrics):
#         super().__init__(dls, model, loss_func=CrossEntropyLoss(), metrics=metrics)

# # Create the model and learner
# num_classes = len(dls.vocab)
# model = ResNetLSTM(num_classes=num_classes)

# learn = CustomLearner(dls, model, metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')])

# # Find the optimal learning rate
# learn.lr_find()

# # Use the optimal learning rate to fine-tune the model
# learn.fit_one_cycle(30, lr_max=1e-3)  # Adjust `lr_max` based on lr_find results

# # Save the entire model using torch.save with .pkl extension
# torch.save(model, './models/full_model_resnet_lstm.pkl')

########################################################################################################################################






























# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch import nn
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy, Precision, Recall, F1Score
# from PIL import Image  # Ensure correct image handling
# import numpy as np
# import torch
# import warnings

# # Suppress all warnings
# warnings.filterwarnings("ignore")

# # Paths
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"

# # Load the dataset
# data = pd.read_csv(LOG_PATH)

# # Ensure the paths to the images are correct
# data['frame_path'] = data['frame_path'].apply(lambda x: os.path.join(FRAME_PATH, x.split('/')[-1]))

# # Define get_x (image path) and get_y (classification labels: keyboard presses)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Custom DataBlock with Data Augmentation
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),
#     item_tfms=Resize(224),
#     batch_tfms=aug_transforms(mult=2, do_flip=True, max_rotate=30.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2)
# )

# # Create the DataLoader
# dls = block.dataloaders(data, bs=8, num_workers=0)

# # Define a simple ResNet-based model (no LSTM for now)
# class SimpleResNet(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleResNet, self).__init__()
#         # Use a ResNet18 model with no pre-trained weights
#         self.resnet = resnet18(weights=None)  # Using weights=None to avoid loading pretrained weights
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Retain until last conv layer
        
#         # Adaptive pooling layer to reduce the spatial dimensions
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # This will output [batch_size, 512, 1, 1]
        
#         # Final fully connected layer
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         # Process through ResNet
#         feature_map = self.resnet(x)
#         pooled_map = self.pool(feature_map)
#         flattened_map = pooled_map.view(x.size(0), -1)
#         output = self.fc(flattened_map)
#         return output

# # Define learner for training the model
# num_classes = len(data['key'].unique())
# model = SimpleResNet(num_classes=num_classes)

# learn = Learner(dls, model, loss_func=CrossEntropyLoss(), metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')])

# # Find the optimal learning rate
# learn.lr_find()

# # Use the optimal learning rate to fine-tune the model
# learn.fit_one_cycle(20, lr_max=1e-3)

# # Save the entire model using torch.save
# torch.save(model, './models/simple_resnet_model.pkl')























# from fastai.vision.all import *
# import pandas as pd
# import os
# from torch.nn import CrossEntropyLoss
# from fastai.metrics import accuracy

# # Load the dataset and adjust the labels to include 'nothing' for 'up'
# LOG_PATH = "./data/dataset.csv"
# FRAME_PATH = "./data/frames/"
# data = pd.read_csv(LOG_PATH)
# data['key'] = data['key'].apply(lambda x: 'nothing' if x == 'up' else x)

# # Define `get_x` (image path) and `get_y` (classification labels: keyboard presses)
# def get_x(row):
#     return row['frame_path']

# def get_y(row):
#     return row['key'].strip()

# # Create a DataBlock for classification task
# block = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),
#     get_x=get_x,
#     get_y=get_y,
#     splitter=RandomSplitter(valid_pct=0.2),
#     item_tfms=Resize(224),
#     batch_tfms=aug_transforms(mult=2)
# )

# # Create the DataLoader
# dls = block.dataloaders(data, bs=16, num_workers=0)

# # Define and train the model
# learn = vision_learner(dls, resnet34, pretrained=False, metrics=[accuracy])
# learn.fine_tune(20)

# # Save the model
# learn.export('./models/model_classification_turns.pkl')
