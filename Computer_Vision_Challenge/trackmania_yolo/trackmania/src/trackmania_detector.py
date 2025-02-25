import os
from ultralytics import YOLO
import cv2
import numpy as np

class TrackmaniaDetector:
    def __init__(self):
        # List of possible locations for the model weights
        possible_paths = [
            os.path.join('wandb', 'latest', 'files', 'best.pt'),
            os.path.join('runs', 'detect', 'train', 'weights', 'best.pt'),
            os.path.join('models', 'best.pt'),
            r'C:\Users\Lenovo\runs\detect\trackmania_model4\weights\best.pt'  # Absolute path to your best.pt
        ]

        # Try to find the existing model weights
        model_path = None
        for path in possible_paths:
            print(f"Checking path: {path}")  # Debug print to check all paths being checked
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path:
            print(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("No trained model found. Using default YOLO model...")
            self.model = YOLO('yolov8n.pt')
            
        # Set confidence threshold
        self.conf_threshold = 0.3
        
    def detect_objects(self, image_path):
        """Detect objects in an image"""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image at {image_path}")
                
            # Run detection with error handling
            try:
                results = self.model(image, conf=self.conf_threshold)
            except Exception as e:
                print(f"Error during detection: {str(e)}")
                return image, []
            
            # Process results
            annotated_image = image.copy()
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, 
                                  (x1, y1), 
                                  (x2, y2), 
                                  (0, 255, 0), 
                                  2)
                    
                    # Add label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_image, 
                                label, 
                                (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 255, 0), 
                                2)
                    
                    # Store detection
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            return annotated_image, detections
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None, []

def main():
    try:
        # Initialize detector
        detector = TrackmaniaDetector()
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
            print("Created 'screenshots' directory. Please place your Trackmania screenshots there.")
            return
        
        # Process all screenshots in the directory
        image_files = [f for f in os.listdir('screenshots') 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images to process")
        
        for filename in image_files:
            image_path = os.path.join('screenshots', filename)
            
            print(f"\nProcessing {filename}...")
            annotated_image, detections = detector.detect_objects(image_path)
            
            if annotated_image is not None:
                # Save annotated image
                output_path = os.path.join('screenshots', f'detected_{filename}')
                cv2.imwrite(output_path, annotated_image)
                
                # Print detected objects
                if detections:
                    print("Detected objects:")
                    for det in detections:
                        print(f"- {det['class']} (confidence: {det['confidence']:.2f})")
                else:
                    print("No objects detected")
                    
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
