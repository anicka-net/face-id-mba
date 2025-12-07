#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QFont, QDesktopServices
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
import random

EMBEDDINGS_FILE = "allowed_faces.npz"
MODEL_REPO = "fal/AuraFace-v1"

SECRETS = [
    "📊 Secret tip: P-value < 0.05 doesn't always mean it matters in business!",
    "💼 MBA wisdom: The best model is the one your stakeholders actually understand",
    "🎓 Fun fact: 80% of data science is data cleaning, 20% is complaining about data cleaning",
    "📈 Insider knowledge: Correlation ≠ Causation (but it makes for great presentations!)",
    "🔮 MBA secret: Excel pivot tables are still more powerful than most people realize",
    "💡 Data truth: Garbage in, garbage out - no ML model can fix bad data collection",
    "🚀 Career hack: Knowing SQL will get you further than most fancy algorithms",
    "🎯 MBA motto: A/B testing beats opinions every time",
    "📉 Reality check: Your neural network probably just needed more training data",
    "🏆 Golden rule: If you can't explain your analysis to management, it's useless"
]

class FaceAuthWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.app = None
        self.init_ui()
        QTimer.singleShot(100, self.init_face_app)
    
    def init_ui(self):
        self.setWindowTitle("Team 04 Authentication System")
        self.setMinimumSize(800, 700)
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Title
        title = QLabel("🔐 TEAM 04 SECURE ACCESS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d4ff; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Status label
        self.status = QLabel("Ready to authenticate")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setFont(QFont("Arial", 16))
        self.status.setStyleSheet("color: #ffffff; margin-bottom: 20px;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        
        # Result label (hidden initially)
        self.result = QLabel()
        self.result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.result.setMinimumHeight(80)
        self.result.setWordWrap(True)
        self.result.hide()
        layout.addWidget(self.result)
        
        # Secret label (hidden initially)
        self.secret = QLabel()
        self.secret.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.secret.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.secret.setStyleSheet("""
            color: #ffd700; 
            margin: 20px;
            background: rgba(255, 215, 0, 0.1);
            border: 2px solid #ffd700;
            border-radius: 10px;
            padding: 25px;
        """)
        self.secret.setWordWrap(True)
        self.secret.hide()
        layout.addWidget(self.secret)
        
        layout.addStretch()
        
        # Authenticate button
        self.auth_btn = QPushButton("🎯 AUTHENTICATE")
        self.auth_btn.setMinimumHeight(100)
        self.auth_btn.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.auth_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auth_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00b4d8, stop:1 #0077b6);
                color: white;
                border: 3px solid #00d4ff;
                border-radius: 15px;
                padding: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0096c7, stop:1 #005f86);
                border: 3px solid #00ffff;
            }
            QPushButton:pressed {
                background: #005f86;
            }
            QPushButton:disabled {
                background: #555555;
                border: 3px solid #777777;
                color: #aaaaaa;
            }
        """)
        self.auth_btn.clicked.connect(self.authenticate)
        layout.addWidget(self.auth_btn)
        
        # Retry button (hidden initially)
        self.retry_btn = QPushButton("🔄 RETRY")
        self.retry_btn.setMinimumHeight(80)
        self.retry_btn.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.retry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retry_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b6b, stop:1 #c92a2a);
                color: white;
                border: 3px solid #ff8787;
                border-radius: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff5252, stop:1 #b71c1c);
                border: 3px solid #ffa8a8;
            }
            QPushButton:pressed {
                background: #b71c1c;
            }
        """)
        self.retry_btn.clicked.connect(self.reset)
        self.retry_btn.hide()
        layout.addWidget(self.retry_btn)
    
    def init_face_app(self):
        try:
            self.status.setText("Loading face recognition model...")
            QApplication.processEvents()
            
            local_dir = snapshot_download(MODEL_REPO, local_dir="models/auraface")
            self.app = FaceAnalysis(
                name="auraface",
                providers=["CPUExecutionProvider"],
                root="."
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.status.setText("Ready to authenticate")
        except Exception as e:
            self.status.setText(f"Error loading model: {str(e)}")
            self.auth_btn.setEnabled(False)
    
    def capture_frame(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
        return frame
    
    def get_face_embedding(self, frame):
        faces = self.app.get(frame)
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding
        return emb.astype("float32")
    
    def load_embeddings(self):
        if not os.path.exists(EMBEDDINGS_FILE):
            return {}, np.zeros((0, 512), dtype="float32")
        data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        names = list(data["names"])
        embs = data["embeddings"].astype("float32")
        name_to_idx = {n: i for i, n in enumerate(names)}
        return name_to_idx, embs
    
    def cosine_similarity(self, a, b):
        return np.dot(b, a)
    
    def authenticate(self):
        if self.app is None:
            self.status.setText("Face recognition not initialized!")
            return
        
        self.auth_btn.setEnabled(False)
        self.status.setText("📸 Capturing image...")
        QApplication.processEvents()
        
        try:
            name_to_idx, embs = self.load_embeddings()
            if len(embs) == 0:
                self.status.setText("❌ No enrolled faces found!")
                self.show_denied()
                return
            
            frame = self.capture_frame()
            self.status.setText("🔍 Analyzing face...")
            QApplication.processEvents()
            
            emb = self.get_face_embedding(frame)
            if emb is None:
                self.status.setText("❌ No face detected!")
                self.show_denied()
                return
            
            sims = self.cosine_similarity(emb, embs)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            
            names = [None] * len(name_to_idx)
            for name, idx in name_to_idx.items():
                names[idx] = name
            best_name = names[best_idx]
            
            threshold = 0.3
            if best_sim >= threshold:
                self.show_granted(best_name, best_sim)
            else:
                self.show_denied(best_name, best_sim)
                
        except Exception as e:
            self.status.setText(f"Error: {str(e)}")
            self.show_denied()
    
    def show_granted(self, name, similarity):
        self.status.setText(f"Welcome, {name}! (Match: {similarity:.1%})")
        self.result.setText("✅ ACCESS GRANTED")
        self.result.setStyleSheet("""
            color: #00ff00;
            background: rgba(0, 255, 0, 0.1);
            border: 3px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
        """)
        self.result.show()
        
        self.secret.setText(f"🔒 TEAM 04 CLASSIFIED INFO:\n\n{random.choice(SECRETS)}")
        self.secret.show()
        
        self.auth_btn.hide()
        self.retry_btn.setText("🏠 RESET")
        self.retry_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #51cf66, stop:1 #2f9e44);
                color: white;
                border: 3px solid #8ce99a;
                border-radius: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #40c057, stop:1 #2b8a3e);
                border: 3px solid #b2f2bb;
            }
            QPushButton:pressed {
                background: #2b8a3e;
            }
        """)
        self.retry_btn.show()
    
    def show_denied(self, name=None, similarity=None):
        if name and similarity:
            self.status.setText(f"Closest match: {name} ({similarity:.1%}) - Not close enough!")
        else:
            self.status.setText("Authentication failed")
        
        self.result.setText("🚫 ACCESS DENIED")
        self.result.setStyleSheet("""
            color: #ff0000;
            background: rgba(255, 0, 0, 0.1);
            border: 3px solid #ff0000;
            border-radius: 10px;
            padding: 20px;
        """)
        self.result.show()
        
        # Rickroll the failed user after a brief delay
        QTimer.singleShot(1500, self.rickroll)
        
        self.auth_btn.hide()
        self.retry_btn.setText("🔄 RETRY")
        self.retry_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b6b, stop:1 #c92a2a);
                color: white;
                border: 3px solid #ff8787;
                border-radius: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff5252, stop:1 #b71c1c);
                border: 3px solid #ffa8a8;
            }
            QPushButton:pressed {
                background: #b71c1c;
            }
        """)
        self.retry_btn.show()
    
    def rickroll(self):
        """Never gonna give you up, never gonna let you down..."""
        QDesktopServices.openUrl(QUrl("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    
    def reset(self):
        self.status.setText("Ready to authenticate")
        self.result.hide()
        self.secret.hide()
        self.retry_btn.hide()
        self.auth_btn.show()
        self.auth_btn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = FaceAuthWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
