import cv2
import os
import json
import numpy as np
import pickle
from deepface import DeepFace
from datetime import datetime

class AttendanceSystem:
    def __init__(self, known_students_dir="known_students", embeddings_cache="embeddings_cache.pkl", distance_threshold=1.0):
        self.known_students_dir = known_students_dir
        self.embeddings_cache = embeddings_cache
        self.distance_threshold = distance_threshold
        self.known_embeddings = []
        self.known_names = []
        
    def load_or_create_embeddings(self):
        """Load embeddings from cache or create new ones from known student images."""
        print("Loading student data...")
        
        if os.path.exists(self.embeddings_cache):
            print("Found cached embeddings, loading...")
            try:
                with open(self.embeddings_cache, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.known_embeddings = cache_data['embeddings']
                    self.known_names = cache_data['names']
                print(f"✓ Loaded {len(self.known_names)} students from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Creating new embeddings...")
                self._create_embeddings()
                self._save_embeddings()
        else:
            print("No cache found, creating new embeddings...")
            self._create_embeddings()
            if self.known_embeddings:
                self._save_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings from images in the known students directory."""
        if not os.path.exists(self.known_students_dir):
            print(f"❌ Directory '{self.known_students_dir}' not found!")
            print("Please create the directory and add student photos.")
            return
        
        image_files = [f for f in os.listdir(self.known_students_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"❌ No image files found in '{self.known_students_dir}'")
            return
        
        print(f"Processing {len(image_files)} student images...")
        successful_loads = 0
        
        for img_file in image_files:
            img_path = os.path.join(self.known_students_dir, img_file)
            try:
                embedding = DeepFace.represent(
                    img_path=img_path, 
                    model_name="Facenet512", 
                    enforce_detection=False
                )[0]["embedding"]
                
                self.known_embeddings.append(np.array(embedding))
                name = os.path.splitext(img_file)[0]
                self.known_names.append(name)
                successful_loads += 1
                print(f"  ✓ {name}")
                
            except Exception as e:
                print(f"  ❌ Failed to process {img_file}: {str(e)[:50]}...")
        
        print(f"Successfully loaded {successful_loads}/{len(image_files)} student images")
    
    def _save_embeddings(self):
        """Save embeddings to cache file."""
        try:
            cache_data = {
                'embeddings': self.known_embeddings,
                'names': self.known_names,
                'created': datetime.now().isoformat(),
                'threshold': self.distance_threshold
            }
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Embeddings cached to {self.embeddings_cache}")
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
    
    def detect_faces(self, image):
        """Detect faces in the image using multiple backends."""
        backends = ["retinaface", "opencv", "mtcnn"]
        
        for backend in backends:
            try:
                detections = DeepFace.extract_faces(
                    img_path=image,
                    enforce_detection=False,
                    detector_backend=backend
                )
                if detections:
                    print(f"Face detection successful using {backend}")
                    return detections
            except Exception as e:
                print(f"  {backend} failed: {str(e)[:30]}...")
                continue
        
        print("❌ All face detection backends failed")
        return []
    
    def recognize_face(self, face_img):
        """Recognize a face by comparing with known embeddings."""
        try:
            face_embedding = DeepFace.represent(
                img_path=face_img, 
                model_name="VGG-Face", 
                enforce_detection=False
            )[0]["embedding"]
            
            face_embedding = np.array(face_embedding)
            
            if len(self.known_embeddings) == 0:
                return "Unknown", float('inf')
            
            # Calculate distances to all known embeddings
            distances = [np.linalg.norm(face_embedding - emb) for emb in self.known_embeddings]
            min_dist = min(distances)
            
            if min_dist < self.distance_threshold:
                idx = distances.index(min_dist)
                return self.known_names[idx], min_dist
            else:
                return "Unknown", min_dist
                
        except Exception:
            return "Unknown", float('inf')
    
    def process_classroom_photo(self, photo_path):
        """Process classroom photo and return attendance results."""
        print(f"\nProcessing classroom photo: {photo_path}")
        
        # Check if photo exists
        if not os.path.exists(photo_path):
            error_msg = f"Photo file '{photo_path}' not found!"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "present": [], "absent": self.known_names}
        
        # Load image
        image = cv2.imread(photo_path)
        if image is None:
            error_msg = f"Could not load image '{photo_path}'. Please check if it's a valid image file."
            print(f"❌ {error_msg}")
            return {"error": error_msg, "present": [], "absent": self.known_names}
        
        print(f"✓ Image loaded successfully ({image.shape[1]}x{image.shape[0]})")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        print("Detecting faces...")
        detections = self.detect_faces(rgb_image)
        
        if not detections:
            print("❌ No faces detected in the image")
            return {
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "present": [],
                "absent": self.known_names,
                "total_faces": 0,
                "message": "No faces detected in the image"
            }
        
        print(f"✓ Found {len(detections)} face(s)")
        detected_students = set()
        recognition_details = []
        
        # Process each detected face
        for i, detection in enumerate(detections, 1):
            face_area = detection.get('facial_area', {})
            x, y, w, h = face_area.get('x', 0), face_area.get('y', 0), face_area.get('w', 0), face_area.get('h', 0)
            
            # Ensure coordinates are within image bounds
            x, y = max(0, x), max(0, y)
            x2, y2 = min(rgb_image.shape[1], x + w), min(rgb_image.shape[0], y + h)
            
            if x2 <= x or y2 <= y:
                print(f"  Face {i}: Invalid coordinates, skipping")
                continue
            
            # Extract face
            face_img = rgb_image[y:y2, x:x2]
            
            # Recognize face
            student_name, confidence = self.recognize_face(face_img)
            
            if student_name != "Unknown":
                detected_students.add(student_name)
                status = "✓ Recognized"
                color = (0, 255, 0)  # Green
            else:
                status = "? Unknown"
                color = (0, 0, 255)  # Red
            
            print(f"  Face {i}: {status} - {student_name} (distance: {confidence:.2f})")
            
            recognition_details.append({
                "face_number": i,
                "name": student_name,
                "confidence_distance": round(confidence, 2),
                "recognized": student_name != "Unknown"
            })
            
            # Draw bounding box and label on image
            cv2.rectangle(image, (x, y), (x2, y2), color, 2)
            label = f"{student_name}"
            if confidence != float('inf'):
                label += f" ({confidence:.1f})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display the result
        self._display_image(image)
        
        # Calculate attendance
        present = sorted(list(detected_students))
        absent = sorted([name for name in self.known_names if name not in present])
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"ATTENDANCE SUMMARY")
        print(f"{'='*50}")
        print(f"Total registered students: {len(self.known_names)}")
        print(f"Faces detected: {len(detections)}")
        print(f"Students present: {len(present)}")
        print(f"Students absent: {len(absent)}")
        
        if present:
            print(f"\nPresent students:")
            for student in present:
                print(f"  ✓ {student}")
        
        if absent:
            print(f"\nAbsent students:")
            for student in absent:
                print(f"  ❌ {student}")
        
        return {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "photo_path": photo_path,
            "total_registered": len(self.known_names),
            "faces_detected": len(detections),
            "present": present,
            "absent": absent,
            "attendance_rate": f"{len(present)}/{len(self.known_names)} ({len(present)/len(self.known_names)*100:.1f}%)",
            "recognition_details": recognition_details
        }
    
    def _display_image(self, image, max_width=1280, max_height=720):
        """Display image scaled to fit screen."""
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h))
        else:
            resized_image = image
        
        window_name = "Attendance Results - Press any key to close"
        cv2.imshow(window_name, resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Configuration
    KNOWN_STUDENTS_DIR = "known_students"
    PHOTO_PATH = "classroom_photo.jpg"  # Change this to your photo filename
    DISTANCE_THRESHOLD = 1.0  # Adjust based on your needs (lower = more strict)
    
    print("="*60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*60)
    
    # Initialize system
    attendance_system = AttendanceSystem(
        known_students_dir=KNOWN_STUDENTS_DIR,
        distance_threshold=DISTANCE_THRESHOLD
    )
    
    # Load student data
    attendance_system.load_or_create_embeddings()
    
    if len(attendance_system.known_names) == 0:
        print("\n❌ No known students loaded!")
        print(f"Please add student photos to the '{KNOWN_STUDENTS_DIR}' directory.")
        print("Each photo should be named with the student's name (e.g., 'john_doe.jpg')")
        return
    
    # Process classroom photo
    results = attendance_system.process_classroom_photo(PHOTO_PATH)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"attendance_results_{timestamp}.json"
    
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")
    
    # Also save a simple CSV for easy viewing
    try:
        csv_file = f"attendance_{timestamp}.csv"
        with open(csv_file, "w") as f:
            f.write("Student Name,Status\n")
            for student in results.get('present', []):
                f.write(f"{student},Present\n")
            for student in results.get('absent', []):
                f.write(f"{student},Absent\n")
        print(f"✓ CSV summary saved to {csv_file}")
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")

if __name__ == "__main__":
    main()