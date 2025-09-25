import cv2
import os
import json
import numpy as np
import pickle
import logging
import sqlite3
import hashlib
import base64
from deepface import DeepFace
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
from pydantic import BaseModel
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attendance_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# Pydantic Models for API
# =========================
class AttendanceResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None
    timestamp: str
    total_registered: int
    faces_detected: int
    students_recognized: int
    present: List[str]
    absent: List[str]
    attendance_rate: str
    recognition_details: List[Dict]
    annotated_image: Optional[str] = None  # Base64 encoded image with annotations

class StudentManualMarkRequest(BaseModel):
    session_id: str
    student_name: str
    action: str  # "mark_present" or "mark_absent"

class AttendanceHistoryResponse(BaseModel):
    sessions: List[Dict]
    total_sessions: int

# =========================
# Database Manager
# =========================
class DatabaseManager:
    def __init__(self, db_path="flutter_attendance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                roll_number TEXT,
                class_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attendance sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                teacher_name TEXT,
                subject TEXT,
                class_name TEXT,
                photo_hash TEXT,
                total_registered INTEGER,
                faces_detected INTEGER,
                students_recognized INTEGER,
                attendance_rate REAL,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attendance records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                student_name TEXT,
                status TEXT CHECK(status IN ('present', 'absent')),
                confidence_score REAL,
                manually_marked BOOLEAN DEFAULT FALSE,
                marked_by TEXT,
                marked_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES attendance_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def save_attendance_session(self, session_data: Dict) -> str:
        """Save attendance session to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            session_id = session_data['session_id']
            
            # Insert session
            cursor.execute('''
                INSERT OR REPLACE INTO attendance_sessions 
                (session_id, teacher_name, subject, class_name, photo_hash, 
                 total_registered, faces_detected, students_recognized, attendance_rate, photo_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                session_data.get('teacher_name', ''),
                session_data.get('subject', ''),
                session_data.get('class_name', ''),
                session_data.get('photo_hash', ''),
                session_data['total_registered'],
                session_data['faces_detected'],
                session_data['students_recognized'],
                float(session_data['attendance_rate'].split('(')[1].split('%')[0]),
                session_data.get('photo_path', '')
            ))
            
            # Clear existing records for this session
            cursor.execute('DELETE FROM attendance_records WHERE session_id = ?', (session_id,))
            
            # Insert attendance records
            for student in session_data['present']:
                cursor.execute('''
                    INSERT INTO attendance_records 
                    (session_id, student_name, status, confidence_score)
                    VALUES (?, ?, 'present', ?)
                ''', (session_id, student, 0.8))  # Default confidence
            
            for student in session_data['absent']:
                cursor.execute('''
                    INSERT INTO attendance_records 
                    (session_id, student_name, status)
                    VALUES (?, ?, 'absent')
                ''', (session_id, student))
            
            conn.commit()
            logger.info(f"Attendance session {session_id} saved successfully")
            return session_id
            
        except Exception as e:
            logger.error(f"Error saving attendance session: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def update_student_status(self, session_id: str, student_name: str, 
                            new_status: str, marked_by: str = "teacher") -> bool:
        """Update student attendance status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE attendance_records 
                SET status = ?, manually_marked = TRUE, marked_by = ?, marked_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND student_name = ?
            ''', (new_status, marked_by, session_id, student_name))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error updating student status: {e}")
            return False
        finally:
            conn.close()
    
    def get_attendance_history(self, limit: int = 20) -> List[Dict]:
        """Get recent attendance sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, teacher_name, subject, class_name, 
                   total_registered, faces_detected, students_recognized,
                   attendance_rate, created_at
            FROM attendance_sessions 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'teacher_name': row[1],
                'subject': row[2],
                'class_name': row[3],
                'total_registered': row[4],
                'faces_detected': row[5],
                'students_recognized': row[6],
                'attendance_rate': f"{row[6]}/{row[4]} ({row[7]:.1f}%)",
                'created_at': row[8]
            })
        
        conn.close()
        return sessions
    
    def get_session_details(self, session_id: str) -> Optional[Dict]:
        """Get detailed session information including all student records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('''
            SELECT * FROM attendance_sessions WHERE session_id = ?
        ''', (session_id,))
        
        session_row = cursor.fetchone()
        if not session_row:
            conn.close()
            return None
        
        # Get attendance records
        cursor.execute('''
            SELECT student_name, status, confidence_score, manually_marked, marked_by, marked_at
            FROM attendance_records 
            WHERE session_id = ?
            ORDER BY student_name
        ''', (session_id,))
        
        records = cursor.fetchall()
        conn.close()
        
        present_students = []
        absent_students = []
        
        for record in records:
            student_info = {
                'name': record[0],
                'confidence_score': record[2] if record[2] else 0.0,
                'manually_marked': bool(record[3]),
                'marked_by': record[4],
                'marked_at': record[5]
            }
            
            if record[1] == 'present':
                present_students.append(student_info)
            else:
                absent_students.append(student_info)
        
        return {
            'session_id': session_row[1],
            'teacher_name': session_row[2],
            'subject': session_row[3],
            'class_name': session_row[4],
            'total_registered': session_row[6],
            'faces_detected': session_row[7],
            'students_recognized': session_row[8],
            'attendance_rate': f"{session_row[8]}/{session_row[6]} ({session_row[9]:.1f}%)",
            'created_at': session_row[11],
            'present_students': present_students,
            'absent_students': absent_students
        }

# =========================
# Enhanced Attendance System
# =========================
class FlutterAttendanceSystem:
    def __init__(self, 
                 known_students_dir="known_students", 
                 embeddings_cache="embeddings_cache.pkl", 
                 distance_threshold=0.8,
                 upload_dir="uploaded_photos"):
        self.known_students_dir = known_students_dir
        self.embeddings_cache = embeddings_cache
        self.distance_threshold = distance_threshold
        self.upload_dir = upload_dir
        self.known_embeddings = []
        self.known_names = []
        self.active_sessions = {}  # Store active sessions in memory
        self.db_manager = DatabaseManager()
        
        # Create directories
        Path(known_students_dir).mkdir(exist_ok=True)
        Path(upload_dir).mkdir(exist_ok=True)
        Path("temp").mkdir(exist_ok=True)
        
        logger.info(f"Flutter Attendance System initialized")
    
    def load_or_create_embeddings(self):
        """Load embeddings from cache or create new ones"""
        logger.info("Loading student embeddings...")
        
        if os.path.exists(self.embeddings_cache):
            try:
                with open(self.embeddings_cache, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.known_embeddings = cache_data['embeddings']
                    self.known_names = cache_data['names']
                logger.info(f"Loaded {len(self.known_names)} students from cache")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self._create_embeddings()
                self._save_embeddings()
        else:
            logger.info("No cache found, creating new embeddings...")
            self._create_embeddings()
            if self.known_embeddings:
                self._save_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings from student images"""
        if not os.path.exists(self.known_students_dir):
            logger.warning(f"Directory '{self.known_students_dir}' not found!")
            return
        
        image_files = [f for f in os.listdir(self.known_students_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            logger.warning(f"No image files found in '{self.known_students_dir}'")
            return
        
        logger.info(f"Processing {len(image_files)} student images...")
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
                name = os.path.splitext(img_file)[0].replace('_', ' ').title()
                self.known_names.append(name)
                successful_loads += 1
                logger.info(f"✓ Processed: {name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {img_file}: {str(e)[:50]}")
        
        logger.info(f"Successfully loaded {successful_loads}/{len(image_files)} student images")
    
    def _save_embeddings(self):
        """Save embeddings to cache"""
        try:
            cache_data = {
                'embeddings': self.known_embeddings,
                'names': self.known_names,
                'created': datetime.now().isoformat(),
                'threshold': self.distance_threshold
            }
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Embeddings cached successfully")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def detect_faces(self, image):
        """Detect faces using multiple backends for better reliability"""
        backends = ["retinaface", "opencv", "mtcnn"]
        
        for backend in backends:
            try:
                detections = DeepFace.extract_faces(
                    img_path=image,
                    enforce_detection=False,
                    detector_backend=backend,
                    align=True
                )
                if detections:
                    logger.info(f"Face detection successful using {backend}")
                    return detections, backend
            except Exception as e:
                logger.warning(f"Face detection failed with {backend}: {str(e)[:50]}")
                continue
        
        logger.error("All face detection backends failed")
        return [], "none"
    
    def recognize_face(self, face_img):
        """Recognize face using embeddings comparison"""
        try:
            face_embedding = DeepFace.represent(
                img_path=face_img, 
                model_name="Facenet512", 
                enforce_detection=False
            )[0]["embedding"]
            
            face_embedding = np.array(face_embedding)
            
            if len(self.known_embeddings) == 0:
                return "Unknown", 0.0
            
            # Calculate distances
            distances = [np.linalg.norm(face_embedding - emb) for emb in self.known_embeddings]
            min_dist = min(distances)
            
            if min_dist < self.distance_threshold:
                idx = distances.index(min_dist)
                confidence = max(0, 1 - (min_dist / self.distance_threshold))
                return self.known_names[idx], confidence
            else:
                return "Unknown", 0.0
                
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return "Unknown", 0.0
    
    def create_annotated_image(self, original_image, detections, recognitions):
        """Create annotated image with bounding boxes and labels"""
        try:
            annotated = original_image.copy()
            
            for i, (detection, recognition) in enumerate(zip(detections, recognitions)):
                name, confidence = recognition
                
                # Get face area coordinates
                face_area = detection.get('facial_area', {})
                x = int(face_area.get('x', 0))
                y = int(face_area.get('y', 0))
                w = int(face_area.get('w', 100))
                h = int(face_area.get('h', 100))
                
                x2, y2 = x + w, y + h
                
                # Choose color based on recognition
                if name != "Unknown":
                    color = (0, 255, 0)  # Green for recognized
                    label = f"{name} ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = "Unknown"
                
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x2, y2), color, 3)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x + 5, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            return original_image
    
    def process_classroom_photo(self, photo_path: str, teacher_name: str = "", 
                              subject: str = "", class_name: str = "") -> Dict:
        """Process classroom photo and return comprehensive attendance results"""
        try:
            # Generate session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(photo_path.encode()).hexdigest()[:8]}"
            
            logger.info(f"Processing classroom photo for session: {session_id}")
            
            # Load and validate image
            image = cv2.imread(photo_path)
            if image is None:
                raise HTTPException(400, "Invalid image file")
            
            # Calculate photo hash
            with open(photo_path, 'rb') as f:
                photo_hash = hashlib.md5(f.read()).hexdigest()
            
            logger.info(f"Image loaded successfully ({image.shape[1]}x{image.shape[0]})")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            logger.info("Detecting faces...")
            detections, detection_backend = self.detect_faces(rgb_image)
            
            if not detections:
                logger.warning("No faces detected in the image")
                result = {
                    'success': False,
                    'message': 'No faces detected in the image',
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'total_registered': len(self.known_names),
                    'faces_detected': 0,
                    'students_recognized': 0,
                    'present': [],
                    'absent': self.known_names.copy(),
                    'attendance_rate': f"0/{len(self.known_names)} (0.0%)",
                    'recognition_details': [],
                    'teacher_name': teacher_name,
                    'subject': subject,
                    'class_name': class_name,
                    'photo_hash': photo_hash
                }
                self.active_sessions[session_id] = result
                return result
            
            logger.info(f"Found {len(detections)} face(s) using {detection_backend}")
            
            # Process each detected face
            detected_students = set()
            recognition_details = []
            recognitions = []
            
            for i, detection in enumerate(detections):
                try:
                    # Extract face region
                    face_area = detection.get('facial_area', {})
                    x, y, w, h = face_area.get('x', 0), face_area.get('y', 0), face_area.get('w', 0), face_area.get('h', 0)
                    
                    # Ensure coordinates are valid
                    x, y = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(rgb_image.shape[1], int(x + w)), min(rgb_image.shape[0], int(y + h))
                    
                    if x2 <= x or y2 <= y:
                        logger.warning(f"Face {i+1}: Invalid coordinates, skipping")
                        recognitions.append(("Unknown", 0.0))
                        continue
                    
                    # Extract and recognize face
                    face_img = rgb_image[y:y2, x:x2]
                    student_name, confidence = self.recognize_face(face_img)
                    recognitions.append((student_name, confidence))
                    
                    if student_name != "Unknown":
                        detected_students.add(student_name)
                        status = "✓ Recognized"
                    else:
                        status = "? Unknown"
                    
                    logger.info(f"Face {i+1}: {status} - {student_name} (confidence: {confidence:.3f})")
                    
                    recognition_details.append({
                        "face_number": i + 1,
                        "name": student_name,
                        "confidence": round(confidence, 3),
                        "recognized": student_name != "Unknown",
                        "coordinates": {"x": x, "y": y, "width": w, "height": h}
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing face {i+1}: {e}")
                    recognitions.append(("Unknown", 0.0))
                    recognition_details.append({
                        "face_number": i + 1,
                        "name": "Error",
                        "confidence": 0.0,
                        "recognized": False,
                        "error": str(e)
                    })
            
            # Create annotated image
            annotated_image = self.create_annotated_image(image, detections, recognitions)
            
            # Encode annotated image to base64
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate attendance
            present = sorted(list(detected_students))
            absent = sorted([name for name in self.known_names if name not in present])
            attendance_rate = len(present) / len(self.known_names) * 100 if self.known_names else 0
            
            # Create comprehensive result
            result = {
                'success': True,
                'message': 'Attendance processed successfully',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'total_registered': len(self.known_names),
                'faces_detected': len(detections),
                'students_recognized': len(present),
                'present': present,
                'absent': absent,
                'attendance_rate': f"{len(present)}/{len(self.known_names)} ({attendance_rate:.1f}%)",
                'recognition_details': recognition_details,
                'annotated_image': annotated_image_b64,
                'detection_backend': detection_backend,
                'teacher_name': teacher_name,
                'subject': subject,
                'class_name': class_name,
                'photo_hash': photo_hash,
                'photo_path': photo_path
            }
            
            # Store session in memory and database
            self.active_sessions[session_id] = result
            
            try:
                self.db_manager.save_attendance_session(result)
                logger.info(f"Session {session_id} saved to database")
            except Exception as e:
                logger.error(f"Failed to save session to database: {e}")
            
            logger.info(f"Attendance processing completed: {len(present)} present, {len(absent)} absent")
            return result
            
        except Exception as e:
            logger.error(f"Error processing classroom photo: {e}")
            raise HTTPException(500, f"Error processing photo: {str(e)}")
    
    def update_student_attendance(self, session_id: str, student_name: str, 
                                action: str, marked_by: str = "teacher") -> Dict:
        """Update student attendance status manually"""
        if session_id not in self.active_sessions:
            # Try to load from database
            session_data = self.db_manager.get_session_details(session_id)
            if not session_data:
                raise HTTPException(404, f"Session {session_id} not found")
            
            # Convert database format to active session format
            self.active_sessions[session_id] = {
                'session_id': session_id,
                'present': [s['name'] for s in session_data['present_students']],
                'absent': [s['name'] for s in session_data['absent_students']],
                'total_registered': session_data['total_registered'],
                'students_recognized': session_data['students_recognized']
            }
        
        session = self.active_sessions[session_id]
        
        if student_name not in self.known_names:
            raise HTTPException(400, f"Student '{student_name}' not registered in the system")
        
        # Update attendance
        if action == "mark_present":
            if student_name in session['absent']:
                session['absent'].remove(student_name)
            if student_name not in session['present']:
                session['present'].append(student_name)
                session['students_recognized'] += 1
        
        elif action == "mark_absent":
            if student_name in session['present']:
                session['present'].remove(student_name)
                session['students_recognized'] -= 1
            if student_name not in session['absent']:
                session['absent'].append(student_name)
        
        else:
            raise HTTPException(400, "Action must be 'mark_present' or 'mark_absent'")
        
        # Update attendance rate
        attendance_rate = len(session['present']) / session['total_registered'] * 100
        session['attendance_rate'] = f"{len(session['present'])}/{session['total_registered']} ({attendance_rate:.1f}%)"
        
        # Update database
        new_status = "present" if action == "mark_present" else "absent"
        success = self.db_manager.update_student_status(session_id, student_name, new_status, marked_by)
        
        if success:
            logger.info(f"Updated {student_name} to {new_status} in session {session_id}")
        else:
            logger.warning(f"Failed to update database for {student_name} in session {session_id}")
        
        return {
            'success': True,
            'message': f"Successfully marked {student_name} as {new_status}",
            'session_id': session_id,
            'student_name': student_name,
            'new_status': new_status,
            'updated_session': session
        }

# =========================
# FastAPI Application
# =========================
app = FastAPI(
    title="Flutter Attendance System API",
    description="Facial Recognition Attendance System API for Flutter Apps",
    version="1.0.0"
)

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize attendance system
attendance_system = FlutterAttendanceSystem()
attendance_system.load_or_create_embeddings()

# =========================
# API Endpoints for Flutter
# =========================

@app.post("/api/process-attendance", response_model=AttendanceResponse)
async def process_attendance(
    photo: UploadFile = File(...),
    teacher_name: str = Form(""),
    subject: str = Form(""),
    class_name: str = Form("")
):
    """
    Main endpoint for Flutter app to upload photo and get attendance results
    """
    try:
        # Validate file
        if not photo.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image (JPEG, PNG)")
        
        # Save uploaded photo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"classroom_{timestamp}_{photo.filename}"
        photo_path = os.path.join(attendance_system.upload_dir, filename)
        
        with open(photo_path, "wb") as f:
            content = await photo.read()
            f.write(content)
        
        logger.info(f"Photo uploaded: {filename} ({len(content)} bytes)")
        
        # Process attendance
        results = attendance_system.process_classroom_photo(
            photo_path, teacher_name, subject, class_name
        )
        
        return AttendanceResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process_attendance: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/api/update-attendance")
async def update_attendance(
    session_id: str = Form(...),
    student_name: str = Form(...),
    action: str = Form(...),  # "mark_present" or "mark_absent"
    marked_by: str = Form("teacher")
):
    """
    Update student attendance status manually
    """
    try:
        result = attendance_system.update_student_attendance(
            session_id, student_name, action, marked_by
        )
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update_attendance: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/api/session/{session_id}")
async def get_session_details(session_id: str):
    """
    Get detailed information about a specific attendance session
    """
    try:
        if session_id in attendance_system.active_sessions:
            return JSONResponse(attendance_system.active_sessions[session_id])
        
        # Try database
        session_data = attendance_system.db_manager.get_session_details(session_id)
        if session_data:
            return JSONResponse(session_data)
        
        raise HTTPException(404, f"Session {session_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session details: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/api/students")
async def get_students():
    """
    Get list of all registered students
    """
    try:
        students = [
            {"name": name, "id": i} 
            for i, name in enumerate(attendance_system.known_names)
        ]
        return JSONResponse({
            "students": students,
            "total_students": len(attendance_system.known_names)
        })
        
    except Exception as e:
        logger.error(f"Error getting students: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/api/history", response_model=AttendanceHistoryResponse)
async def get_attendance_history(limit: int = Query(20, ge=1, le=100)):
    """
    Get attendance history for Flutter app
    """
    try:
        sessions = attendance_system.db_manager.get_attendance_history(limit)
        return AttendanceHistoryResponse(
            sessions=sessions,
            total_sessions=len(sessions)
        )
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/api/system-status")
async def get_system_status():
    """
    Get system status and configuration for Flutter app
    """
    try:
        status = {
            "status": "online",
            "registered_students": len(attendance_system.known_names),
            "distance_threshold": attendance_system.distance_threshold,
            "face_recognition_model": "Facenet512",
            "face_detection_backends": ["retinaface", "opencv", "mtcnn"],
            "active_sessions": len(attendance_system.active_sessions),
            "system_time": datetime.now().isoformat(),
            "cache_exists": os.path.exists(attendance_system.embeddings_cache),
            "upload_directory": attendance_system.upload_dir,
            "known_students_directory": attendance_system.known_students_dir
        }
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/api/add-student")
async def add_student(
    student_photo: UploadFile = File(...),
    student_name: str = Form(...),
    roll_number: str = Form(""),
    class_name: str = Form("")
):
    """
    Add a new student to the system (for teachers to register new students)
    """
    try:
        # Validate inputs
        if not student_photo.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        if not student_name.strip():
            raise HTTPException(400, "Student name is required")
        
        # Check if student already exists
        if student_name in attendance_system.known_names:
            raise HTTPException(400, f"Student '{student_name}' already exists")
        
        # Save photo
        clean_name = student_name.replace(' ', '_').lower()
        filename = f"{clean_name}.jpg"
        photo_path = os.path.join(attendance_system.known_students_dir, filename)
        
        with open(photo_path, "wb") as f:
            content = await student_photo.read()
            f.write(content)
        
        # Add student to system
        success = attendance_system.add_student(student_name, photo_path)
        
        if success:
            logger.info(f"New student added: {student_name}")
            return JSONResponse({
                "success": True,
                "message": f"Student '{student_name}' added successfully",
                "student_name": student_name,
                "total_students": len(attendance_system.known_names)
            })
        else:
            # Remove the saved photo if adding failed
            if os.path.exists(photo_path):
                os.remove(photo_path)
            raise HTTPException(400, "Failed to process student photo. Please ensure the photo contains a clear face.")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding student: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.delete("/api/clear-session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a specific session from memory (keep in database)
    """
    try:
        if session_id in attendance_system.active_sessions:
            del attendance_system.active_sessions[session_id]
            return JSONResponse({
                "success": True,
                "message": f"Session {session_id} cleared from memory"
            })
        else:
            raise HTTPException(404, f"Session {session_id} not found in active sessions")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/api/export-attendance/{session_id}")
async def export_attendance(session_id: str, format: str = Form("json")):
    """
    Export attendance data in different formats (JSON, CSV)
    """
    try:
        # Get session data
        session_data = None
        if session_id in attendance_system.active_sessions:
            session_data = attendance_system.active_sessions[session_id]
        else:
            session_data = attendance_system.db_manager.get_session_details(session_id)
        
        if not session_data:
            raise HTTPException(404, f"Session {session_id} not found")
        
        if format.lower() == "csv":
            # Create CSV format
            csv_content = "Student Name,Status,Confidence,Manually Marked\n"
            
            # Add present students
            for student in session_data.get('present', []):
                csv_content += f"{student},Present,N/A,No\n"
            
            # Add absent students  
            for student in session_data.get('absent', []):
                csv_content += f"{student},Absent,N/A,No\n"
            
            return JSONResponse({
                "success": True,
                "format": "csv",
                "session_id": session_id,
                "content": csv_content,
                "filename": f"attendance_{session_id}.csv"
            })
        
        else:  # Default to JSON
            return JSONResponse({
                "success": True,
                "format": "json",
                "session_id": session_id,
                "content": session_data,
                "filename": f"attendance_{session_id}.json"
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting attendance: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/api/attendance-summary")
async def get_attendance_summary(days: int = Query(7, ge=1, le=30)):
    """
    Get attendance summary for the last N days
    """
    try:
        sessions = attendance_system.db_manager.get_attendance_history(100)  # Get more to filter by date
        
        # Filter by days
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = [
            session for session in sessions 
            if datetime.fromisoformat(session['created_at'].replace('Z', '')) > cutoff_date
        ]
        
        # Calculate summary statistics
        total_sessions = len(recent_sessions)
        total_students = len(attendance_system.known_names)
        
        if total_sessions == 0:
            return JSONResponse({
                "summary": {
                    "period_days": days,
                    "total_sessions": 0,
                    "average_attendance_rate": 0.0,
                    "total_registered_students": total_students,
                    "most_active_day": "N/A",
                    "sessions": []
                }
            })
        
        # Calculate average attendance rate
        attendance_rates = []
        for session in recent_sessions:
            rate_str = session['attendance_rate'].split('(')[1].split('%')[0]
            attendance_rates.append(float(rate_str))
        
        avg_attendance = sum(attendance_rates) / len(attendance_rates)
        
        # Find most active day (most sessions)
        day_counts = {}
        for session in recent_sessions:
            date_str = session['created_at'].split('T')[0]
            day_counts[date_str] = day_counts.get(date_str, 0) + 1
        
        most_active_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else "N/A"
        
        return JSONResponse({
            "summary": {
                "period_days": days,
                "total_sessions": total_sessions,
                "average_attendance_rate": round(avg_attendance, 1),
                "total_registered_students": total_students,
                "most_active_day": most_active_day,
                "sessions": recent_sessions[:10]  # Return latest 10 sessions
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting attendance summary: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

# =========================
# Utility Endpoints
# =========================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return JSONResponse({
        "message": "Flutter Attendance System API",
        "version": "1.0.0",
        "endpoints": {
            "process_attendance": "/api/process-attendance",
            "update_attendance": "/api/update-attendance",
            "get_students": "/api/students",
            "get_history": "/api/history",
            "system_status": "/api/system-status",
            "add_student": "/api/add-student",
            "health_check": "/api/health"
        },
        "documentation": "/docs"
    })

# =========================
# Enhanced Attendance System Methods
# =========================

def add_student_method(self, name: str, image_path: str) -> bool:
    """Add a new student to the system"""
    try:
        # Process the image and create embedding
        result = DeepFace.represent(
            img_path=image_path, 
            model_name="Facenet512", 
            enforce_detection=False,
            detector_backend="retinaface"
        )
        
        if result and len(result) > 0:
            embedding = result[0]["embedding"]
            self.known_embeddings.append(np.array(embedding))
            self.known_names.append(name)
            
            # Update cache
            self._save_embeddings()
            
            logger.info(f"Successfully added new student: {name}")
            return True
        else:
            logger.warning(f"No face detected in image for student: {name}")
            return False
            
    except Exception as e:
        logger.error(f"Error adding student {name}: {e}")
        return False

# Add the method to the class
FlutterAttendanceSystem.add_student = add_student_method

# =========================
# Error Handlers
# =========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# =========================
# Main Application Runner
# =========================

if __name__ == "__main__":
    print("="*60)
    print("🎓 FLUTTER ATTENDANCE SYSTEM API")
    print("="*60)
    print("📱 Designed for Flutter mobile applications")
    print("📁 Student photos directory: known_students/")
    print("📸 Upload photos as: student_name.jpg")
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("🚀 Health Check: http://localhost:8000/api/health")
    print("="*60)
    
    # Startup message
    total_students = len(attendance_system.known_names)
    print(f"✅ System initialized with {total_students} registered students")
    
    if total_students == 0:
        print("⚠️  Warning: No students registered!")
        print("   Add student photos to 'known_students/' directory")
        print("   Or use the /api/add-student endpoint")
    
    print("\n🎯 Key API Endpoints for Flutter:")
    print("   POST /api/process-attendance - Upload photo & get attendance")
    print("   POST /api/update-attendance - Mark student present/absent")
    print("   GET  /api/students - Get all registered students")
    print("   GET  /api/history - Get attendance history")
    print("   GET  /api/system-status - Get system status")
    print("   POST /api/add-student - Register new student")
    
    print("\n🔄 Starting server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )