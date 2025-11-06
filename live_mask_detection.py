import cv2
import numpy as np

class RuleBasedMaskDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.cap = self.initialize_camera()
        
    def initialize_camera(self):
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"✅ Camera opened at index {i}")
                return cap
        return None

    def detect_mask_rule_based(self, face_roi):
        """Enhanced rule-based mask detection using color and texture analysis"""
        try:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            height, width = face_roi.shape[:2]
            
            # Analyze lower half of face (where masks are worn)
            lower_face = face_roi[height//2:, :]
            
            if lower_face.size == 0:
                return "No Face", 0.0
            
            # Convert to different color spaces for better analysis
            lower_face_hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
            lower_face_lab = cv2.cvtColor(lower_face, cv2.COLOR_BGR2LAB)
            
            # Common mask colors (blue, white, black, green surgical masks)
            mask_color_ranges = [
                # Blue masks
                ([100, 50, 50], [130, 255, 255]),
                # White masks  
                ([0, 0, 200], [180, 30, 255]),
                # Black masks
                ([0, 0, 0], [180, 50, 100]),
                # Green surgical masks
                ([35, 50, 50], [85, 255, 255])
            ]
            
            total_mask_pixels = 0
            total_pixels = lower_face.shape[0] * lower_face.shape[1]
            
            for lower, upper in mask_color_ranges:
                mask = cv2.inRange(lower_face_hsv, np.array(lower), np.array(upper))
                total_mask_pixels += np.sum(mask) / 255
            
            mask_coverage = total_mask_pixels / total_pixels
            
            # Additional check: texture analysis (masks have less texture than skin)
            gray_lower_face = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            texture_variance = np.var(gray_lower_face)
            
            # Decision logic
            if mask_coverage > 0.4:  # High mask-like color coverage
                confidence = min(90.0, 60 + (mask_coverage * 50))
                return "Mask", confidence
            elif mask_coverage > 0.25:  # Moderate coverage
                # Check if it might be skin (more texture)
                if texture_variance > 500:  # Skin has more texture
                    return "No Mask", 70.0
                else:
                    return "Mask", 65.0
            else:
                # Low coverage, likely no mask
                confidence = min(85.0, 70 + ((1 - mask_coverage) * 30))
                return "No Mask", confidence
                
        except Exception as e:
            return "Error", 0.0

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        return faces

    def run_detection(self):
        if self.cap is None:
            print("❌ No camera found")
            return
            
        print("🎯 Starting Rule-Based Mask Detection")
        print("💡 Using color and texture analysis only")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            faces = self.detect_faces(frame)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Extract face region
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        label, confidence = self.detect_mask_rule_based(face_roi)
                        
                        # Set color based on label
                        if label == "Mask":
                            color = (0, 255, 0)  # Green
                        elif label == "No Mask":
                            color = (0, 0, 255)  # Red
                        else:
                            color = (255, 165, 0)  # Orange
                        
                        # Draw results
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                        text = f"{label} ({confidence:.1f}%) [Rule-Based]"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "No Face Detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame, "Rule-Based Detection Only", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("🎭 Rule-Based Mask Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Detection stopped")

# Run the detector
if __name__ == "__main__":
    detector = RuleBasedMaskDetector()
    detector.run_detection()
