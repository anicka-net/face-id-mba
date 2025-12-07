#!/usr/bin/env python3
"""
AuraFace Enrollment Tool s vizuálním náhledem
Umožňuje přidat, smazat a zobrazit enrolled faces
"""

import sys
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download

EMBEDDINGS_FILE = "allowed_faces.npz"
MODEL_REPO = "fal/AuraFace-v1"

def load_embeddings():
    """Načte uložené embeddingy"""
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}, np.zeros((0, 512), dtype="float32")
    
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    names = list(data["names"])
    embs = data["embeddings"].astype("float32")
    name_to_idx = {n: i for i, n in enumerate(names)}
    return name_to_idx, embs

def save_embeddings(name_to_idx, embs):
    """Uloží embeddingy do souboru"""
    names = [None] * len(name_to_idx)
    for name, idx in name_to_idx.items():
        names[idx] = name
    
    np.savez(EMBEDDINGS_FILE, names=np.array(names), embeddings=embs)

def list_enrolled():
    """Zobrazí seznam enrolled faces"""
    name_to_idx, embs = load_embeddings()
    
    if len(embs) == 0:
        print("No enrolled faces")
        return
    
    print(f"Enrolled faces ({len(embs)}):")
    for name in sorted(name_to_idx.keys()):
        print(f"   - {name}")

def remove_face(name):
    """Smaže enrolled face"""
    name_to_idx, embs = load_embeddings()
    
    if name not in name_to_idx:
        print(f"'{name}' not found in database")
        return
    
    idx = name_to_idx[name]
    
    # Odstraň z name_to_idx
    del name_to_idx[name]
    
    # Odstraň z embeddings
    embs = np.delete(embs, idx, axis=0)
    
    # Přeindexuj
    new_name_to_idx = {}
    for n, old_idx in name_to_idx.items():
        if old_idx > idx:
            new_name_to_idx[n] = old_idx - 1
        else:
            new_name_to_idx[n] = old_idx
    
    save_embeddings(new_name_to_idx, embs)
    print(f"'{name}' removed")

def enroll_face(app, name, multi_capture=False):
    """Enrollment s live preview"""
    print(f"\n{'='*60}")
    print(f"Enrolling: {name}")
    print(f"{'='*60}")
    
    if multi_capture:
        print("Multi-capture mode: Taking 3 photos")
        print("   (for better accuracy)")
    else:
        print("Single-capture mode")
    
    print("\nControls:")
    print("  SPACE - Capture face")
    print("  ESC   - Cancel")
    print()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Nelze otevřít kameru")
        return False
    
    # Nastavení rozlišení (větší obraz)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    captured_embeddings = []
    captures_needed = 3 if multi_capture else 1
    captures_done = 0
    
    print(f"Camera ready! Position yourself...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detekuj obličeje
        faces = app.get(frame)
        
        # Vytvoř display frame
        display = frame.copy()
        
        if len(faces) > 0:
            # Najdi největší obličej
            faces_sorted = sorted(faces, 
                                 key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), 
                                 reverse=True)
            largest_face = faces_sorted[0]
            
            # Nakresluj čtvereček kolem obličeje
            bbox = largest_face.bbox.astype(int)
            
            # Zelený čtvereček, tlustší čára
            cv2.rectangle(display, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         (0, 255, 0), 
                         3)
            
            # Přidej rohové markery (vypadá to cool 😎)
            corner_length = 20
            thickness = 4
            color = (0, 255, 0)
            
            # Levý horní roh
            cv2.line(display, (bbox[0], bbox[1]), (bbox[0] + corner_length, bbox[1]), color, thickness)
            cv2.line(display, (bbox[0], bbox[1]), (bbox[0], bbox[1] + corner_length), color, thickness)
            
            # Pravý horní roh
            cv2.line(display, (bbox[2], bbox[1]), (bbox[2] - corner_length, bbox[1]), color, thickness)
            cv2.line(display, (bbox[2], bbox[1]), (bbox[2], bbox[1] + corner_length), color, thickness)
            
            # Levý dolní roh
            cv2.line(display, (bbox[0], bbox[3]), (bbox[0] + corner_length, bbox[3]), color, thickness)
            cv2.line(display, (bbox[0], bbox[3]), (bbox[0], bbox[3] - corner_length), color, thickness)
            
            # Pravý dolní roh
            cv2.line(display, (bbox[2], bbox[3]), (bbox[2] - corner_length, bbox[3]), color, thickness)
            cv2.line(display, (bbox[2], bbox[3]), (bbox[2], bbox[3] - corner_length), color, thickness)
            
            # Status text na pozadí
            status_text = "FACE DETECTED"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(display, 
                         (bbox[0], bbox[1] - text_size[1] - 15),
                         (bbox[0] + text_size[0] + 10, bbox[1] - 5),
                         (0, 255, 0),
                         -1)
            cv2.putText(display, status_text,
                       (bbox[0] + 5, bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 0), 2)
        else:
            # Žádný obličej - varování
            cv2.putText(display, "No face detected",
                       (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 0, 255), 2)
        
        # Instrukce dole
        if multi_capture:
            instruction = f"Photo {captures_done + 1}/{captures_needed} - SPACE: Capture | ESC: Cancel"
        else:
            instruction = "SPACE: Capture | ESC: Cancel"
        
        # Tmavé pozadí pro text
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(display,
                     (10, display.shape[0] - text_size[1] - 20),
                     (text_size[0] + 20, display.shape[0] - 10),
                     (0, 0, 0),
                     -1)
        
        cv2.putText(display, instruction,
                   (15, display.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        
        # Ukaž náhled
        cv2.imshow(f"Enrollment - {name}", display)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            print("Enrollment cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        elif key == 32:  # SPACE
            if len(faces) == 0:
                print("No face detected, try again")
                continue
            
            # Vezmi největší obličej
            faces_sorted = sorted(faces,
                                 key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                                 reverse=True)
            
            emb = faces_sorted[0].normed_embedding.astype("float32")
            captured_embeddings.append(emb)
            captures_done += 1
            
            # Flash efekt (bílý)
            white = np.ones_like(display) * 255
            cv2.imshow(f"Enrollment - {name}", white)
            cv2.waitKey(100)
            
            print(f"Photo {captures_done}/{captures_needed} captured")
            
            if captures_done < captures_needed:
                if multi_capture:
                    print(f"  Turn your head slightly and press SPACE again...")
            else:
                # Done!
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(captured_embeddings) == 0:
        print("No photos captured")
        return False
    
    # Process embeddings
    if len(captured_embeddings) > 1:
        print(f"\nAveraging {len(captured_embeddings)} photos...")
        final_embedding = np.mean(captured_embeddings, axis=0)
        # Renormalize
        final_embedding = final_embedding / np.linalg.norm(final_embedding)
    else:
        final_embedding = captured_embeddings[0]
    
    # Load existing embeddings
    name_to_idx, embs = load_embeddings()
    
    # Check if already exists
    if name in name_to_idx:
        print(f"'{name}' already exists, updating...")
        idx = name_to_idx[name]
        embs[idx] = final_embedding
    else:
        print(f"Adding new entry '{name}'")
        name_to_idx[name] = len(embs)
        embs = np.vstack([embs, final_embedding.reshape(1, -1)])
    
    # Save
    save_embeddings(name_to_idx, embs)
    
    print(f"\n{'='*60}")
    print(f"Enrollment complete!")
    print(f"{'='*60}\n")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("AuraFace Enrollment Tool")
        print("\nUsage:")
        print(f"  {sys.argv[0]} add <name>           - Add face (1 photo)")
        print(f"  {sys.argv[0]} add <name> --multi   - Add face (3 photos, better)")
        print(f"  {sys.argv[0]} list                 - Show enrolled faces")
        print(f"  {sys.argv[0]} remove <name>        - Remove face")
        print("\nExamples:")
        print(f"  {sys.argv[0]} add Anicka")
        print(f"  {sys.argv[0]} add Anicka --multi")
        print(f"  {sys.argv[0]} list")
        print(f"  {sys.argv[0]} remove Anicka")
        return 1
    
    command = sys.argv[1]
    
    if command == "list":
        list_enrolled()
        return 0
    
    if command == "remove":
        if len(sys.argv) < 3:
            print("❌ Zadej jméno k odstranění")
            return 1
        remove_face(sys.argv[2])
        return 0
    
    if command == "add":
        if len(sys.argv) < 3:
            print("Error: Please provide a name")
            return 1
        
        name = sys.argv[2]
        multi = len(sys.argv) > 3 and sys.argv[3] == "--multi"
        
        print("Loading AuraFace model...")
        try:
            local_dir = snapshot_download(MODEL_REPO, local_dir="models/auraface")
            app = FaceAnalysis(
                name="auraface",
                providers=["CPUExecutionProvider"],
                root="."
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("Model loaded\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
        
        success = enroll_face(app, name, multi)
        return 0 if success else 1
    
    print(f"Unknown command: {command}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
