# Face Authentication System

A modern face recognition system using state-of-the-art AuraFace deep learning model for secure access control.

## 🎯 Project Overview

This project implements a facial authentication system with:

- **Enrollment Tool**: Interactive face capture with live preview and multi-photo support
- **Authentication GUI**: Visual feedback with real-time face detection
- **Diagnostic Tool**: Test and analyze similarity scores
- **Persistent Storage**: Secure face embedding database

Built with AuraFace (InsightFace), PyQt6, and OpenCV.

## 📋 System Requirements

- **Operating System**: Linux (Ubuntu/Debian, Fedora, openSUSE), macOS, or Windows
- **Camera**: Webcam or built-in camera
- **Display**: 1280x720 or higher recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: ~500MB for models and dependencies
- **Python**: 3.8 or higher

---

## 🔧 Installation

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libopencv-dev \
    python3-opencv
```

**Fedora:**
```bash
sudo dnf install -y \
    python3 \
    python3-pip \
    opencv \
    python3-opencv
```

**openSUSE Tumbleweed:**
```bash
sudo zypper install -y \
    python3 \
    python3-pip \
    opencv \
    python3-opencv
```

**macOS (Homebrew):**
```bash
brew install python opencv
```

**Windows:**
```powershell
# Install Python from python.org
# OpenCV will be installed via pip
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install opencv-python numpy insightface onnxruntime huggingface-hub PyQt6
```

**Required packages:**
- `opencv-python` (≥4.0): Computer vision and image processing
- `numpy` (≥1.19): Numerical computing
- `insightface` (≥0.7): Face analysis and AuraFace model
- `onnxruntime` (≥1.12): ONNX model inference engine
- `huggingface-hub` (≥0.14): Model downloading from Hugging Face
- `PyQt6` (≥6.0): GUI framework for main application

### Step 4: Verify Installation

```bash
python3 -c "import cv2, numpy, insightface, PyQt6; print('✅ All dependencies installed successfully!')"
```

If this prints the success message, you're ready to go!

---

## 🚀 Quick Start

### 1. Enroll Your Face

**Simple enrollment (1 photo):**
```bash
python3 enroll_auraface.py add YourName
```

**Better enrollment (3 photos, recommended):**
```bash
python3 enroll_auraface.py add YourName --multi
```

**During enrollment:**
- Position your face in the green rectangle
- Ensure good, even lighting
- Look directly at the camera
- Press **SPACE** when ready to capture
- For multi-capture mode: slightly turn your head between photos

**Manage enrolled faces:**
```bash
# List all enrolled faces
python3 enroll_auraface.py list

# Remove a face
python3 enroll_auraface.py remove YourName
```

### 2. Test Recognition (Recommended)

Before using the main app, test your enrollment:

```bash
python3 test_recognition.py
```

This opens a diagnostic window showing:
- Real-time face detection with green boxes
- Similarity scores for each detected face
- Live feedback on match quality

Press **SPACE** for detailed analysis, **ESC** to quit.

### 3. Run Authentication System

```bash
python3 face_mba.py
```

The main application will open with a modern interface. Click the **🎯 AUTHENTICATE** button to test face recognition.

**On successful authentication:**
- Shows "✅ ACCESS GRANTED"
- Displays a random data science/MBA wisdom
- Welcome message with match percentage

**On failed authentication:**
- Shows "🚫 ACCESS DENIED"
- Opens rickroll in browser (easter egg!)
- Option to retry

---

## 📊 Understanding Similarity Scores

The system uses cosine similarity (0.0 to 1.0) displayed as percentages:

| Score Range | Interpretation | Typical Result |
|-------------|----------------|----------------|
| **85-100%** | Same person, ideal conditions | ✅ GRANTED |
| **70-85%** | Same person, slight variations | ✅ GRANTED (if threshold allows) |
| **50-70%** | Ambiguous match | ⚠️ Usually DENIED |
| **30-50%** | Different person | ❌ DENIED |
| **0-30%** | Very different person | ❌ DENIED |

### Adjusting the Threshold

The default threshold is **0.3 (30%)**, which is quite permissive. For better security:

**Edit `face_mba.py`:**
```python
THRESHOLD = 0.3  # Change to 0.5, 0.6, or even 0.7 for stricter matching
```

**Recommended thresholds:**
- **0.5-0.6**: Balanced security and usability
- **0.7-0.8**: High security, may occasionally reject valid users
- **0.3-0.4**: Low security, easier to authenticate

After changing the threshold, restart the application.

---

## 🔍 Troubleshooting

### Installation Issues

**Problem: `ModuleNotFoundError: No module named 'insightface'`**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install missing package
pip install insightface onnxruntime
```

**Problem: `ImportError: libGL.so.1: cannot open shared object file`**
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# Fedora
sudo dnf install mesa-libGL
```

**Problem: Model download fails**
```bash
# Check internet connection
# If in restricted region, try mirror:
export HF_ENDPOINT=https://hf-mirror.com
python3 enroll_auraface.py add Test
```

### Camera Issues

**Problem: Camera not detected**
```bash
# Test camera with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('✅ OK' if cap.isOpened() else '❌ FAIL')"

# Try different camera index (if you have multiple cameras)
# Edit scripts and change VideoCapture(0) to VideoCapture(1) or VideoCapture(2)
```

**Problem: Camera permissions denied (Linux)**
```bash
# Add your user to video group
sudo usermod -aG video $USER

# Log out and log back in for changes to take effect
```

### Recognition Issues

**Problem: Everyone gets authenticated (false positives)**

This means the threshold is too low or enrollment quality is poor.

**Solution 1: Increase threshold**
```python
# In face_mba.py, change:
THRESHOLD = 0.6  # or 0.7 for even stricter
```

**Solution 2: Re-enroll with better quality**
```bash
# Remove old enrollment
python3 enroll_auraface.py remove YourName

# Re-enroll with multi-capture and good lighting
python3 enroll_auraface.py add YourName --multi
```

**Tips for better enrollment:**
- Use bright, even lighting (avoid strong shadows)
- Remove glasses during enrollment if possible
- Keep face centered and at medium distance
- Use multi-capture mode for robustness

**Problem: Never authenticates (false negatives)**

Threshold might be too high or lighting conditions changed.

**Solution:**
```python
# Lower threshold in face_mba.py
THRESHOLD = 0.4  # or 0.5

# Or re-enroll in current lighting conditions
```

**Problem: No face detected during enrollment**
- Move closer to the camera
- Improve lighting
- Ensure face is visible and unobstructed
- Try different camera angle

---

## 📁 Project Structure

```
face-authentication-system/
│
├── face_mba.py                  # Main authentication GUI application
├── enroll_auraface.py          # Face enrollment tool with live preview
├── test_recognition.py          # Diagnostic tool for testing
│
├── allowed_faces.npz           # Stored face embeddings (generated)
├── models/                     # Downloaded AuraFace models (auto-created)
│   └── auraface/
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # License file
```

---

## 🎓 How It Works

### Architecture

**AuraFace Pipeline:**

```
1. Camera Input (640×480 or higher)
   ↓
2. Face Detection (InsightFace detector)
   ↓
3. Face Alignment & Preprocessing
   ↓
4. AuraFace CNN Feature Extraction (ResNet-based)
   ↓
5. 512-Dimensional Embedding (L2-normalized)
   ↓
6. Cosine Similarity Comparison
   ↓
7. Threshold Decision → GRANT/DENY
```

### Technical Details

- **Model**: AuraFace v1 (from Hugging Face: `fal/AuraFace-v1`)
- **Architecture**: ResNet-based CNN with ArcFace loss
- **Embedding Size**: 512 dimensions (normalized to unit sphere)
- **Face Detection**: RetinaFace-based detector (InsightFace)
- **Similarity Metric**: Cosine similarity (dot product of normalized vectors)
- **Inference Time**: 
  - CPU: 200-500ms per authentication
  - GPU: 20-50ms per authentication (if available)
- **Model Size**: ~100-200 MB
- **Training Data**: Millions of face images (not disclosed)
- **Accuracy**: 95-99% on standard benchmarks (LFW, CFP-FP)

### Why AuraFace?

- **State-of-the-art accuracy**: Trained on massive datasets
- **Robust to variations**: Handles pose, lighting, age, expression changes
- **Compact embeddings**: 512-D is efficient yet powerful
- **Production-ready**: Used in commercial applications
- **Open source**: Available via InsightFace framework

### Cosine Similarity Explained

Given two face embeddings **A** and **B** (both 512-D unit vectors):

```
Similarity = A · B = Σ(A[i] × B[i])  for i = 0 to 511
```

Since both vectors are normalized (length = 1):
```
Similarity = cos(θ)
```

Where **θ** is the angle between vectors in 512-dimensional space.

- **Similarity = 1.0**: Vectors point in same direction (0° angle) → Identical
- **Similarity = 0.9**: Small angle (~25°) → Very similar
- **Similarity = 0.5**: Large angle (60°) → Different
- **Similarity = 0.0**: Perpendicular (90°) → Completely different

---

## 🔒 Security Considerations

**⚠️ Important:** This is a demonstration/educational project. For production deployment, implement:

### Critical Security Enhancements Needed

1. **Liveness Detection**
   - Detect photos/videos (presentation attacks)
   - Blink detection, motion analysis, texture analysis
   - Depth sensing (if available)

2. **Secure Storage**
   - Encrypt face embeddings at rest
   - Use secure key management
   - Hash or encrypt usernames

3. **Audit & Logging**
   - Log all authentication attempts
   - Track failed attempts (rate limiting)
   - Alert on suspicious activity

4. **Network Security** (if deployed remotely)
   - Use HTTPS/TLS for all communication
   - Implement proper authentication tokens
   - Secure API endpoints

5. **Multi-Factor Authentication**
   - Combine with password, PIN, or hardware token
   - Face recognition as 2FA, not sole method

6. **Privacy Compliance**
   - GDPR, CCPA compliance for biometric data
   - User consent mechanisms
   - Right to deletion

7. **Model Security**
   - Regular updates to latest models
   - Monitor for adversarial attacks
   - Implement model versioning

### Known Limitations

- ❌ **No liveness detection**: Can be fooled by photos/videos
- ❌ **Plaintext embeddings**: Stored without encryption
- ❌ **No presentation attack detection**: Vulnerable to deepfakes
- ❌ **Limited to frontal faces**: Best results with face-to-camera
- ❌ **Single-factor auth**: No additional verification
- ❌ **No audit trail**: Authentication events not logged

**Use this project for:**
- ✅ Learning and education
- ✅ Prototyping
- ✅ Internal testing
- ✅ Controlled environments

**Do NOT use for:**
- ❌ Production security systems
- ❌ Financial applications
- ❌ Medical records access
- ❌ Critical infrastructure

---

## 📚 Dependencies

### Core Dependencies

```txt
opencv-python>=4.5.0
numpy>=1.19.0
insightface>=0.7.0
onnxruntime>=1.12.0
huggingface-hub>=0.14.0
PyQt6>=6.0.0
```

### Optional Dependencies

```txt
onnxruntime-gpu>=1.12.0  # For GPU acceleration (CUDA required)
```

### Creating requirements.txt

```bash
# Generate from current environment
pip freeze > requirements.txt

# Or use provided minimal requirements
cat > requirements.txt << EOF
opencv-python>=4.5.0
numpy>=1.19.0
insightface>=0.7.0
onnxruntime>=1.12.0
huggingface-hub>=0.14.0
PyQt6>=6.0.0
EOF

# Install from requirements.txt
pip install -r requirements.txt
```

---

## 🎯 Usage Examples

### Example 1: Enroll Multiple People

```bash
# Enroll team members
python3 enroll_auraface.py add Alice --multi
python3 enroll_auraface.py add Bob --multi
python3 enroll_auraface.py add Charlie --multi

# Verify enrollments
python3 enroll_auraface.py list

# Output:
# Enrolled faces (3):
#    - Alice
#    - Bob
#    - Charlie
```

### Example 2: Test Before Deployment

```bash
# Start diagnostic tool
python3 test_recognition.py

# For each person:
# 1. Have them stand in front of camera
# 2. Press SPACE to test
# 3. Verify scores are correct:
#    - Own face: 70-95%
#    - Other faces: 20-50%
```

### Example 3: Adjust for Your Use Case

**High Security (office access, secure facility):**
```python
# face_mba.py
THRESHOLD = 0.75  # Very strict
```

**Balanced (personal project, home automation):**
```python
# face_mba.py
THRESHOLD = 0.55  # Balanced
```

**Convenience (friendly demo, low stakes):**
```python
# face_mba.py
THRESHOLD = 0.35  # Permissive
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Liveness detection implementation
- [ ] Database backend (SQLite, PostgreSQL)
- [ ] REST API for remote authentication
- [ ] Web interface (Flask/FastAPI)
- [ ] Mobile app (React Native, Flutter)
- [ ] GPU acceleration optimization
- [ ] Docker containerization
- [ ] Unit tests and CI/CD
- [ ] Performance benchmarking suite
- [ ] Multi-language support

**How to contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### Third-Party Licenses

- **InsightFace/AuraFace**: MIT License
- **OpenCV**: Apache 2.0 License
- **PyQt6**: GPL v3 / Commercial License
- **NumPy**: BSD License
- **ONNX Runtime**: MIT License

---

## 👥 Credits

**Original Implementation:**
- Team 04

**Enhancements:**
- Anicka - Enrollment tool with live preview, diagnostic tools, documentation

**Built With:**
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis framework
- [AuraFace](https://huggingface.co/fal/AuraFace-v1) - Face recognition model
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [ONNX Runtime](https://onnxruntime.ai/) - Inference engine

---

## 🙏 Acknowledgments

- InsightFace team for excellent face recognition models
- Hugging Face for model hosting
- OpenCV community for computer vision tools
- PyQt team for cross-platform GUI framework

---

## 📞 Support

**Having issues?**

1. Check the [Troubleshooting](#-troubleshooting) section
2. Verify all dependencies are installed: `python3 -c "import cv2, insightface, PyQt6"`
3. Test camera separately: `python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
4. Check Python version: `python3 --version` (need 3.8+)
5. Ensure virtual environment is activated

**Common issues:**
- **Camera not working**: Check permissions, try different index
- **Everyone authenticated**: Increase threshold, re-enroll with better lighting
- **Nobody authenticated**: Decrease threshold, check enrollment quality
- **Slow inference**: Normal on CPU, consider GPU acceleration

---

## 🗺️ Roadmap

**Version 1.1 (Planned):**
- [ ] Liveness detection (basic blink detection)
- [ ] SQLite database backend
- [ ] Configuration file support
- [ ] Improved error messages
- [ ] Unit tests

**Version 2.0 (Future):**
- [ ] Web interface with REST API
- [ ] Advanced liveness detection
- [ ] Multi-face simultaneous recognition
- [ ] Admin dashboard
- [ ] Docker deployment
- [ ] Mobile app

**Version 3.0 (Vision):**
- [ ] Cloud deployment guide (AWS, GCP, Azure)
- [ ] Kubernetes support
- [ ] Advanced analytics
- [ ] Federated learning support
- [ ] Edge device optimization

---

## 📈 Performance Tips

### Improve Speed

**Enable GPU acceleration (if NVIDIA GPU available):**
```bash
# Uninstall CPU version
pip uninstall onnxruntime

# Install GPU version
pip install onnxruntime-gpu

# Edit face_mba.py and enroll_auraface.py:
# Change providers=["CPUExecutionProvider"]
# To providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
```

**Reduce camera resolution:**
```python
# In scripts, modify:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower from 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower from 720
```

### Improve Accuracy

**Multi-capture enrollment:**
```bash
# Always use --multi flag
python3 enroll_auraface.py add Name --multi
```

**Good lighting:**
- Use even, frontal lighting
- Avoid strong shadows
- Natural light or soft LED lights work best

**Camera positioning:**
- Face camera directly (within ±15°)
- Medium distance (arm's length)
- Eye level height

**Multiple enrollments:**
```bash
# Enroll same person in different conditions
python3 enroll_auraface.py add Name_morning --multi  # Morning light
python3 enroll_auraface.py add Name_evening --multi  # Evening light
# System will treat as different entries but can be merged
```

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Python**: 3.8+  
**License**: MIT
