from ultralytics import YOLO
import yaml
import os
import shutil
import random

def create_val_set(train_path, val_path, split_ratio=0.2):
    """Create validation set from training data"""
    # Create validation directories if they don't exist
    os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)
    
    # Get list of training images
    train_images = [f for f in os.listdir(os.path.join(train_path, 'images'))
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Calculate number of validation images
    num_val = max(1, int(len(train_images) * split_ratio))
    
    # Randomly select images for validation
    val_images = random.sample(train_images, num_val)
    
    # Move selected images and their labels to validation set
    for img_name in val_images:
        # Move image
        src_img = os.path.join(train_path, 'images', img_name)
        dst_img = os.path.join(val_path, 'images', img_name)
        shutil.copy2(src_img, dst_img)
        
        # Move corresponding label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(train_path, 'labels', label_name)
        dst_label = os.path.join(val_path, 'labels', label_name)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    return len(val_images)

def setup_training():
    # Get absolute path of the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define Trackmania-specific classes
    classes = [
        'car',
        'wheel',
        'road',
        'barrier',
        'checkpoint',
        'finish_line',
        'tree',
        'building'
    ]
    
    # Create absolute paths
    dataset_path = os.path.join(project_dir, 'dataset')
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    # Create train directory if it doesn't exist
    os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
    
    # Verify training images exist
    train_images = [f for f in os.listdir(os.path.join(train_path, 'images'))
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(train_images) == 0:
        print("Error: No training images found!")
        return False
    
    print(f"Found {len(train_images)} training images")
    
    # Create validation set if it doesn't exist
    val_images = []
    if not os.path.exists(os.path.join(val_path, 'images')):
        print("Creating validation set from training data...")
        num_val = create_val_set(train_path, val_path)
        print(f"Created validation set with {num_val} images")
    
    # Create dataset config
    data = {
        'path': dataset_path,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    # Save dataset config
    config_path = os.path.join(project_dir, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(data, f)
    
    return True

def train_model():
    # Get absolute path of the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_dir, 'dataset.yaml')
    
    # Load a pre-trained YOLO model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=config_path,
        epochs=30,
        imgsz=640,
        batch=16,
        name='trackmania_model'
    )
    
    return results

if __name__ == "__main__":
    print("Setting up training configuration...")
    if setup_training():
        print("\nStarting model training...")
        train_model()
    else:
        print("\nPlease ensure:")
        print("1. Your dataset directory structure is:")
        print("   dataset/")
        print("   ├── train/")
        print("   │   ├── images/")
        print("   │   └── labels/")
        print("   └── val/")
        print("       ├── images/")
        print("       └── labels/")
        print("2. You have training images in dataset/train/images/")
        print("3. You have corresponding labels in dataset/train/labels/")