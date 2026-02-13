"""
UNIVERSITY SMART ATTENDANCE SYSTEM
Entry & Exit Tracking with Face Recognition
Uses DeepFace with VGG-Face for extreme precision
"""
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from datetime import datetime
import csv
import os
import tempfile

# Settings
DATABASE_FILE = "ultra_database.pkl"
ATTENDANCE_FILE = "attendance_log.csv"
COOLDOWN_TRACKER_FILE = "cooldown_tracker.pkl"
SIMILARITY_THRESHOLD = 0.35
CONFIDENCE_MARGIN = 0.12

# Attendance Mode Selection
print("="*70)
print("   UNIVERSITY SMART ATTENDANCE SYSTEM")
print("   Face Recognition - Entry & Exit Tracking")
print("="*70)
print("\nSelect Mode:")
print("  [1] 📥 ENTRY - Students entering (Check-in)")
print("  [2] 📤 EXIT - Students leaving (Check-out)")
print("="*70)

mode_choice = input("\nEnter mode (1 or 2): ").strip()

if mode_choice == "1":
    ATTENDANCE_MODE = "ENTRY"
    mode_color = (0, 255, 0)  # Green
    mode_emoji = "📥"
elif mode_choice == "2":
    ATTENDANCE_MODE = "EXIT"
    mode_color = (0, 165, 255)  # Orange
    mode_emoji = "📤"
else:
    print("Invalid choice. Defaulting to ENTRY mode.")
    ATTENDANCE_MODE = "ENTRY"
    mode_color = (0, 255, 0)
    mode_emoji = "📥"

# Load database with error handling
if not os.path.exists(DATABASE_FILE):
    print("✗ Database not found!")
    print("\nPlease run training first:")
    print("  python ultra_train.py")
    print("\nThen run this again.")
    input("\nPress Enter to exit...")
    exit()

try:
    with open(DATABASE_FILE, 'rb') as f:
        database = pickle.load(f)
    if not database or len(database) == 0:
        print("✗ Database is empty!")
        print("\nPlease run:")
        print("  1. python ultra_register.py (register people)")
        print("  2. python ultra_train.py (train)")
        print("  3. python ultra_attendance.py (this)")
        input("\nPress Enter to exit...")
        exit()
except Exception as e:
    print(f"✗ Error loading database: {e}")
    print("\nPlease run training again:")
    print("  python ultra_train.py")
    input("\nPress Enter to exit...")
    exit()

print("="*70)
print(f"   UNIVERSITY ATTENDANCE - {ATTENDANCE_MODE} MODE {mode_emoji}")
print("   Deep Learning Recognition (VGG-Face)")
print("="*70)
print(f"✓ Database loaded successfully!")
print(f"✓ Registered students: {len(database)}")

# Show who is registered
print("\nRegistered students:")
for person_id, person_data in database.items():
    print(f"  - {person_data['name']} (ID: {person_id})")

print(f"\nMode: {ATTENDANCE_MODE}")
print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
print(f"Confidence margin: {CONFIDENCE_MARGIN}")
print("\nFeatures:")
print("  ✓ Background-independent")
print("  ✓ Works in different locations")
print("  ✓ Won't mix similar students")
print(f"  ✓ Logging {ATTENDANCE_MODE} times")
print("\nPress ESC to exit")
print("="*70 + "\n")

# Create attendance file
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Name', 'Date', 'Time', 'Type', 'Similarity', 'Model'])

# Load cooldown tracker (remembers last login times across sessions)
last_logged_time = {}
cooldown_file_mode = f"cooldown_{ATTENDANCE_MODE.lower()}.pkl"

if os.path.exists(cooldown_file_mode):
    try:
        with open(cooldown_file_mode, 'rb') as f:
            last_logged_time = pickle.load(f)
        print(f"✓ Loaded previous {ATTENDANCE_MODE} session data")

        # Show who is still in cooldown
        current_time = datetime.now()
        in_cooldown = []
        for person_id, last_time in last_logged_time.items():
            time_since = (current_time - last_time).total_seconds() / 60
            if time_since < COOLDOWN_MINUTES:
                person_name = database[person_id]['name'] if person_id in database else person_id
                remaining = COOLDOWN_MINUTES - time_since
                in_cooldown.append(f"  - {person_name}: {remaining:.1f} min remaining")

        if in_cooldown:
            print(f"\n⏱ Students in {ATTENDANCE_MODE} cooldown:")
            for msg in in_cooldown:
                print(msg)
        print()
    except Exception as e:
        print(f"⚠ Could not load cooldown data: {e}")
        last_logged_time = {}

logged_today = set()
COOLDOWN_MINUTES = 5  # Cooldown period in minutes

# Open camera
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_skip = 0  # Process every 10th frame (deep learning is slow)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_skip += 1
    display = frame.copy()

    # Process every 10 frames (deep learning is computationally expensive)
    if frame_skip % 10 == 0:
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Extract face with padding reduction to minimize background
            # Add small margin but crop tighter to avoid background
            margin = 10  # Small margin (was implicitly larger before)

            # Calculate crop coordinates with margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            # Extract face region
            face_img = frame[y1:y2, x1:x2]

            # Resize to standard size to reduce background influence
            face_img = cv2.resize(face_img, (224, 224))  # Standard size

            # Save to temp file for DeepFace
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, face_img)

            try:
                # Generate embedding using DeepFace
                test_embedding = DeepFace.represent(
                    img_path=temp_file.name,
                    model_name="VGG-Face",
                    enforce_detection=False
                )[0]["embedding"]

                # Find best match
                best_match_id = None
                best_similarity = float('inf')
                second_best_similarity = float('inf')

                for person_id, person_data in database.items():
                    # Calculate cosine distance
                    stored_embedding = person_data['embedding']

                    # Cosine distance
                    distance = np.linalg.norm(np.array(test_embedding) - np.array(stored_embedding))

                    if distance < best_similarity:
                        second_best_similarity = best_similarity
                        best_similarity = distance
                        best_match_id = person_id
                    elif distance < second_best_similarity:
                        second_best_similarity = distance

                # Check if match is good enough AND significantly better than second best
                # This helps distinguish between similar people (siblings, twins)
                margin = second_best_similarity - best_similarity

                if best_similarity < SIMILARITY_THRESHOLD and margin > CONFIDENCE_MARGIN:
                    # MATCH FOUND with high confidence!
                    person_name = database[best_match_id]['name']

                    # Check cooldown
                    current_time = datetime.now()
                    can_log = True
                    remaining = 0

                    if best_match_id in last_logged_time:
                        time_since_last = (current_time - last_logged_time[best_match_id]).total_seconds() / 60
                        if time_since_last < COOLDOWN_MINUTES:
                            can_log = False
                            remaining = COOLDOWN_MINUTES - time_since_last

                    # Visual feedback - ALWAYS show recognition
                    if can_log:
                        color = mode_color  # Use mode color (green for entry, orange for exit)
                        status = f"READY {mode_emoji}"
                    else:
                        color = (128, 128, 128)  # Gray - cooldown
                        status = "COOLDOWN"

                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)

                    label = f"{person_name} [{status}]"
                    cv2.putText(display, label, (x, y-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    similarity_pct = (1 - best_similarity) * 100
                    cv2.putText(display, f"Match: {similarity_pct:.1f}%", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Debug: Show confidence margin
                    cv2.putText(display, f"Confidence: {margin:.3f}", (x, y+h+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Show cooldown timer if in cooldown
                    if not can_log:
                        cv2.putText(display, f"Next log in: {remaining:.1f} min",
                                   (x, y+h+25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Log attendance ONLY if cooldown passed
                    if can_log:
                        with open(ATTENDANCE_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                best_match_id,
                                person_name,
                                current_time.strftime("%Y-%m-%d"),
                                current_time.strftime("%H:%M:%S"),
                                ATTENDANCE_MODE,  # ENTRY or EXIT
                                f"{similarity_pct:.2f}%",
                                "VGG-Face"
                            ])

                        logged_today.add(best_match_id)
                        last_logged_time[best_match_id] = current_time

                        # Save cooldown tracker to file (persistent across sessions)
                        try:
                            with open(cooldown_file_mode, 'wb') as f:
                                pickle.dump(last_logged_time, f)
                        except Exception as e:
                            print(f"⚠ Warning: Could not save cooldown data: {e}")

                        print(f"✓ {ATTENDANCE_MODE}: {person_name} (Similarity: {similarity_pct:.1f}%, Confidence: {margin:.3f})")
                    else:
                        # Still recognized, just not logged
                        print(f"↻ Recognized: {person_name} (Cooldown: {remaining:.1f} min remaining, Margin: {margin:.3f})")

                else:
                    # NO MATCH - Unknown person or not confident enough
                    color = (0, 0, 255)  # Red
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

                    if best_similarity < SIMILARITY_THRESHOLD:
                        # Close match but not confident (too similar to multiple people)
                        cv2.putText(display, "UNCERTAIN - Too Similar", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(display, f"Margin too low: {margin:.3f}", (x, y+h+25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        print(f"⚠ UNCERTAIN: Best={best_similarity:.3f}, 2nd={second_best_similarity:.3f}, Margin={margin:.3f} (need >{CONFIDENCE_MARGIN})")
                    else:
                        # No good match at all
                        cv2.putText(display, "UNKNOWN", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        similarity_pct = (1 - best_similarity) * 100
                        cv2.putText(display, f"Best match: {similarity_pct:.1f}%",
                                   (x, y+h+25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                # Error in recognition
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(display, "Processing...", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    # Draw UI
    cv2.putText(display, f"{ATTENDANCE_MODE} Mode - Logged: {len(logged_today)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    cv2.putText(display, "University Attendance System (VGG-Face)", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(f"University Attendance - {ATTENDANCE_MODE} Mode - ESC to exit", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()

print(f"\n✓ {ATTENDANCE_MODE} session ended")
print(f"  Total logged: {len(logged_today)}")
print(f"  Attendance file: {ATTENDANCE_FILE}")