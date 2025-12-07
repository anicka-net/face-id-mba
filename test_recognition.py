#!/usr/bin/env python3
"""
Face Recognition Diagnostic Tool for AuraFace
Tests similarity scores between enrolled faces and live camera
"""

import sys
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

EMBEDDINGS_FILE = "allowed_faces.npz"
THRESHOLD = 0.3  # Original threshold from your app

def load_embeddings():
    """Load enrolled face embeddings"""
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}, np.zeros((0, 512), dtype="float32")
    
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    names = list(data["names"])
    embs = data["embeddings"].astype("float32")
    name_to_idx = {n: i for i, n in enumerate(names)}
    return name_to_idx, embs

def cosine_similarity(a, b):
    """Calculate cosine similarity (dot product for normalized vectors)"""
    return np.dot(b, a)

def get_face_embedding(app, frame):
    """Extract face embedding from frame"""
    faces = app.get(frame)
    if len(faces) == 0:
        return None
    
    # Get largest face
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].normed_embedding
    return emb.astype("float32")

def draw_face_boxes(frame, app, name_to_idx, embs):
    """Draw bounding boxes with similarity scores"""
    faces = app.get(frame)
    
    for face in faces:
        # Draw bounding box - thicker lines for better visibility
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        
        # Calculate similarities
        emb = face.normed_embedding.astype("float32")
        
        if len(embs) > 0:
            sims = cosine_similarity(emb, embs)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            
            names = [None] * len(name_to_idx)
            for name, idx in name_to_idx.items():
                names[idx] = name
            best_name = names[best_idx]
            
            # Display name and similarity - smaller text
            text = f"{best_name}: {int(best_sim * 100)}%"
            color = (0, 255, 0) if best_sim >= THRESHOLD else (0, 0, 255)
            
            # Add background for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame,
                         (bbox[0], bbox[1] - text_size[1] - 10),
                         (bbox[0] + text_size[0] + 8, bbox[1] - 2),
                         color,
                         -1)
            
            cv2.putText(frame, text, (bbox[0] + 4, bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame

def main():
    print("=" * 60)
    print("AuraFace Recognition Diagnostic Tool")
    print("=" * 60)
    print()
    
    # Initialize face analysis
    print("Loading AuraFace model...")
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download("fal/AuraFace-v1", local_dir="models/auraface")
        
        app = FaceAnalysis(
            name="auraface",
            providers=["CPUExecutionProvider"],
            root="."
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    print()
    
    # Load enrolled faces
    name_to_idx, embs = load_embeddings()
    
    if len(embs) == 0:
        print("❌ No faces enrolled!")
        print("   Run the main app and enroll faces first.")
        return 1
    
    print(f"✅ Found {len(embs)} enrolled face(s):")
    for name in name_to_idx.keys():
        print(f"   - {name}")
    print()
    
    # Calculate similarity between enrolled faces
    if len(embs) > 1:
        print("📊 Similarity between enrolled faces:")
        names = [None] * len(name_to_idx)
        for name, idx in name_to_idx.items():
            names[idx] = name
        
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = cosine_similarity(embs[i], embs[j])
                print(f"   {names[i]} ↔ {names[j]}: {sim:.4f} ({int(sim * 100)}%)")
        print()
    
    # Open camera
    print("📸 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return 1
    
    # Set higher resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Camera opened")
    print()
    print("Controls:")
    print("  SPACE - Test current frame (detailed output)")
    print("  ESC   - Quit")
    print()
    
    # Main loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw face boxes and similarities
            display_frame = draw_face_boxes(frame.copy(), app, name_to_idx, embs)
            
            # Add instructions - smaller text with background
            instruction_text = "SPACE: Test | ESC: Quit"
            text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Dark background for instructions
            cv2.rectangle(display_frame,
                         (10, 10),
                         (text_size[0] + 30, text_size[1] + 25),
                         (0, 0, 0),
                         -1)
            
            cv2.putText(display_frame, instruction_text, 
                       (20, text_size[1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Create named window with specific size
            cv2.namedWindow("AuraFace Recognition Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AuraFace Recognition Test", 1280, 720)
            cv2.imshow("AuraFace Recognition Test", display_frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                print("-" * 60)
                emb = get_face_embedding(app, frame)
                
                if emb is None:
                    print("⚠️  No face detected")
                    print()
                    continue
                
                # Calculate similarities with all enrolled faces
                sims = cosine_similarity(emb, embs)
                
                print("📊 Similarity scores:")
                names = [None] * len(name_to_idx)
                for name, idx in name_to_idx.items():
                    names[idx] = name
                
                # Sort by similarity
                sorted_indices = np.argsort(sims)[::-1]
                
                for idx in sorted_indices:
                    sim = sims[idx]
                    name = names[idx]
                    
                    status = ""
                    if sim >= THRESHOLD:
                        status = "✅ MATCH"
                    elif sim >= THRESHOLD * 0.8:
                        status = "⚠️  WEAK"
                    else:
                        status = "❌ NO MATCH"
                    
                    print(f"   {name:20s}: {sim:.4f} ({int(sim * 100):3d}%)  {status}")
                
                print()
                
                # Show recommendation
                best_sim = float(np.max(sims))
                if best_sim >= THRESHOLD:
                    print(f"✅ Would GRANT access (score {best_sim:.4f} >= threshold {THRESHOLD})")
                else:
                    print(f"❌ Would DENY access (score {best_sim:.4f} < threshold {THRESHOLD})")
                    gap = THRESHOLD - best_sim
                    print(f"   Gap to threshold: {gap:.4f} ({int(gap * 100)}%)")
                
                print()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print()
    print("💡 Tips for better accuracy:")
    print("   - Same person should score 0.70-0.95")
    print("   - Different person should score 0.10-0.50")
    print("   - If scores overlap, try:")
    print("     • Better lighting when enrolling")
    print("     • Enroll multiple times from different angles")
    print("     • Increase threshold in face_mba.py")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
