import cv2
import os
import json
from datetime import datetime

class ImageLabeler:
    def __init__(self, image_dir='screenshots'):
        self.image_dir = image_dir
        self.current_image = None
        self.current_image_name = None
        self.labels = []
        self.classes = [
            'car',
            'wheel',
            'road',
            'barrier',
            'checkpoint',
            'finish_line',
            'tree',
            'building'
        ]
        # Initialize states
        self.drawing = False
        self.selecting_class = False
        self.selected_label_idx = -1
        self.ix, self.iy = -1, -1
        self.current_class = 0
        self.temp_class = 0
        
    def start_labeling(self):
        images = [f for f in os.listdir(self.image_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"No images found in {self.image_dir}")
            return
            
        print("\nLabeling Instructions:")
        print("1. Left click and drag to draw bounding box")
        print("2. Press number keys (1-8) to cycle through classes")
        print("3. Press Enter to confirm class selection")
        print("4. Click on existing box to change its class")
        print("5. Press 'n' for next image")
        print("6. Press 'q' to quit")
        print("7. Press 'd' to delete selected box\n")
        
        for i, cls in enumerate(self.classes, 1):
            print(f"   {i}: {cls}")
        
        # Create window with normal size
        cv2.namedWindow('Labeler', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Labeler', self.mouse_callback)
        
        for image_name in images:
            self.current_image_name = image_name
            self.current_image = cv2.imread(os.path.join(self.image_dir, image_name))
            
            if self.current_image is None:
                print(f"Failed to load image: {image_name}")
                continue
                
            # Resize window to fit screen while maintaining aspect ratio
            screen_height = 900  # Maximum height
            height, width = self.current_image.shape[:2]
            scale = min(screen_height / height, 1.0)
            window_width = int(width * scale)
            window_height = int(height * scale)
            cv2.resizeWindow('Labeler', window_width, window_height)
            
            self.labels = []
            print(f"\nLabeling image: {image_name}")
            
            while True:
                img_copy = self.current_image.copy()
                
                # Draw existing labels
                for i, label in enumerate(self.labels):
                    x1, y1, x2, y2 = label['bbox']
                    
                    # If this label is being edited, show temporary class
                    if i == self.selected_label_idx and self.selecting_class:
                        cls_name = self.classes[self.temp_class]
                        color = (255, 0, 0)  # Blue for editing
                    else:
                        cls_name = self.classes[label['class']]
                        color = (0, 255, 0)  # Green for confirmed
                        
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_copy, cls_name, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw current box while dragging
                if self.drawing:
                    cv2.rectangle(img_copy, (self.ix, self.iy), 
                                (self.current_x, self.current_y), (255, 0, 0), 2)
                    # Show current class while drawing
                    cls_name = self.classes[self.temp_class]
                    cv2.putText(img_copy, cls_name, (self.ix, self.iy-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                cv2.imshow('Labeler', img_copy)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    self.save_labels()
                    break
                elif key == ord('d') and self.selected_label_idx != -1:
                    # Delete selected box
                    self.labels.pop(self.selected_label_idx)
                    self.selected_label_idx = -1
                    self.selecting_class = False
                    print("Deleted selected box")
                elif key in [ord(str(i)) for i in range(1, len(self.classes) + 1)]:
                    # Update temporary class
                    self.temp_class = int(chr(key)) - 1
                    print(f"Selected class: {self.classes[self.temp_class]}")
                elif key == 13:  # Enter key
                    if self.selecting_class:
                        # Confirm class change for existing box
                        self.labels[self.selected_label_idx]['class'] = self.temp_class
                        print(f"Changed class to: {self.classes[self.temp_class]}")
                        self.selecting_class = False
                        self.selected_label_idx = -1
        
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
        self.current_x = x
        self.current_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on existing box
            for i, label in enumerate(self.labels):
                x1, y1, x2, y2 = label['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_label_idx = i
                    self.selecting_class = True
                    self.temp_class = label['class']
                    print(f"Editing box with class: {self.classes[self.temp_class]}")
                    return
            
            # If not clicking on existing box, start drawing new one
            self.drawing = True
            self.ix, self.iy = x, y
            self.temp_class = self.current_class
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = min(self.ix, x), min(self.iy, y)
                x2, y2 = max(self.ix, x), max(self.iy, y)
                
                # Only add if box has some size
                if x1 != x2 and y1 != y2:
                    self.labels.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': self.temp_class
                    })
                    print(f"Added {self.classes[self.temp_class]} box")
    
    def save_labels(self):
        if not self.labels:
            return
            
        # Convert to YOLO format
        img_height, img_width = self.current_image.shape[:2]
        
        label_dir = os.path.join('dataset', 'train', 'labels')
        os.makedirs(label_dir, exist_ok=True)
        
        base_name = os.path.splitext(self.current_image_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            for label in self.labels:
                x1, y1, x2, y2 = label['bbox']
                # Convert to YOLO format (normalized coordinates)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                f.write(f"{label['class']} {x_center} {y_center} {width} {height}\n")
        
        # Copy image to dataset
        img_dir = os.path.join('dataset', 'train', 'images')
        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, self.current_image_name), self.current_image)
        print(f"Saved labels for {self.current_image_name}")

if __name__ == "__main__":
    labeler = ImageLabeler()
    labeler.start_labeling()