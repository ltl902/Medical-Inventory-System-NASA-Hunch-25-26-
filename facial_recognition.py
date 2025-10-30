import logging
import sys, os
from contextlib import contextmanager

# Fix for Qt platform plugin errors (e.g. "Could not find the Qt platform plugin 'wayland'")
# Must set before importing cv2 (which may initialize Qt plugins).
# Prefer XCB (widely available on X11 systems); fall back to 'offscreen' if that fails.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
try:
    # Some environments prefer an explicit plugin path — leave unset by default.
    pass
except Exception:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import re
import numpy as np
import time
import threading
import queue
import insightface
import gc  #  Added for garbage collection

logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

#  Context manager to silence native output (C/C++ level)
@contextmanager
def suppress_native_output():
    try:
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(orig_stdout_fd, 1)
        os.dup2(orig_stderr_fd, 2)
        os.close(devnull)
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)


def safe_exit(cap=None):
    """Close the camera and OpenCV windows safely."""
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Camera safely closed.")


def main():
    # -----------------------------
    # LOAD INSIGHTFACE MODEL (quiet)
    # -----------------------------
    with suppress_native_output():
        app = insightface.app.FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

    # -----------------------------
    # HELPER FUNCTIONS
    # -----------------------------
    def normalize(emb):
        return emb / np.linalg.norm(emb)

    def distance_to_confidence(dist, max_dist=2.0):
        return max(0.0, min(1.0, 1.0 - dist / max_dist))

    # -----------------------------
    # LOAD REFERENCES
    # -----------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(script_dir, "references")

    if not os.path.exists(ref_dir):
        print("Reference folder not found!")
       
        return 3

    reference_embeddings = {}

    for filename in os.listdir(ref_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(ref_dir, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read {filename}")
                return 2

            faces = app.get(img)
            if len(faces) == 0:
                print(f"Warning: No face detected in {filename}")
                return 3

            embedding = normalize(faces[0].embedding)
            label = os.path.splitext(filename)[0]
            label = re.sub(r'\d+$', '', label)
            label = re.sub(r'[^A-Za-z0-9_\-]', '_', label)

            if label not in reference_embeddings:
                reference_embeddings[label] = []
            reference_embeddings[label].append(embedding)

    for label in reference_embeddings:
        reference_embeddings[label] = np.array(reference_embeddings[label])

    if not reference_embeddings:
        print("No valid reference faces loaded.")
        return []

    THRESHOLD = 1

    def recognize_face(face_embedding):
        best_match = None
        best_score = float("inf")
        for label, embeddings in reference_embeddings.items():
            distances = np.linalg.norm(embeddings - face_embedding, axis=1)
            score = np.min(distances)
            if score < best_score:
                best_score = score
                best_match = label
        if best_score < THRESHOLD:
            return best_match, best_score
        else:
            return "Unknown", best_score

    # -----------------------------
    # THREADING SETUP
    # -----------------------------
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    detected_names = set()
    known_face_detected = False  # 🔹 Only close after detecting known face

    def recognition_worker():
        while not stop_event.is_set():
            if frame_queue.empty():
                time.sleep(0.01)
                continue
            frame = frame_queue.get()
            faces = app.get(frame)
            results = []
            for face in faces:
                emb = normalize(face.embedding)
                name, dist = recognize_face(emb)
                confidence = distance_to_confidence(dist)
                box = face.bbox.astype(int)
                results.append((box, name, confidence))
            if not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except Exception:
                    pass
            result_queue.put(results)
            time.sleep(0.02)

    # -----------------------------
    # START WEBCAM & THREAD
    # -----------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        safe_exit(cap)
        print("Webcam could not be opened.")
        return 4

    thread = threading.Thread(target=recognition_worker, daemon=True)
    thread.start()
    
    print("Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()
    last_results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            frame_count += 1

            # 🕐 Only process every 5th frame
            if frame_count % 75 == 0 and frame_queue.empty():
                frame_queue.put(frame.copy())

            if not result_queue.empty():
                last_results = result_queue.get()

            # Draw faces and handle first detections
            for box, name, confidence in last_results:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                text = f"{name} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, text, (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Print new known faces only once
                if name != "Unknown" and name not in detected_names:
                    detected_names.add(name)
                    print(f"Person detected: {name}")
                    known_face_detected = True

            # FPS display
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)

            # Garbage collection every 200 frames
            if frame_count % 200 == 0:
                gc.collect()

            # Exit loop if a known face was detected
            if known_face_detected:
                stop_event.set()
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break

    except Exception as e:
        print(f"Error during webcam processing: {e}")

    safe_exit(cap)
    return list(detected_names)


if __name__ == "__main__":
    try:
        detected = main()
        print("Program finished. People seen:", detected)
    except Exception as e:
        print(f" Error in facial recognition: {e}")