"""
ULTRA-ACCURATE FACE REGISTRATION
Uses DeepFace with VGG-Face model for extreme accuracy
Can distinguish twins!
"""

import cv2
import os
from deepface import DeepFace
import numpy as np
import pickle

# Settings
DATABASE_PATH = "face_database"
SAMPLES_PER_PERSON = 10  # Increased to 10 for better accuracy

os.makedirs(DATABASE_PATH, exist_ok=True)

print("="*70)
print("ULTRA-ACCURATE FACE REGISTRATION")
print("Uses Deep Learning (VGG-Face)")
print("="*70)

person_name = input("\nEnter person's name: ").strip()
person_id = input("Enter unique ID (1, 2, 3...): ").strip()

person_folder = os.path.join(DATABASE_PATH, f"{person_id}_{person_name}")
os.makedirs(person_folder, exist_ok=True)

print(f"\nRegistering: {person_name} (ID: {person_id})")
print(f"Will capture {SAMPLES_PER_PERSON} high-quality photos")
print("\nIMPORTANT for twins/similar people:")
print("  - Use DIFFERENT expressions (smile, neutral, serious)")
print("  - Different angles (left, front, right)")
print("  - Different lighting positions")
print("\nPress SPACE to capture each photo")
print("Press ESC to cancel\n")

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

captured = 0
embeddings = []

while captured < SAMPLES_PER_PERSON:
    ret, frame = cam.read()
    if not ret:
        break

    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw instructions
    display = frame.copy()

    # Updated instructions for 10 photos
    if captured == 0:
        instruction = "Photo 1: Look STRAIGHT, NEUTRAL expression"
    elif captured == 1:
        instruction = "Photo 2: Look STRAIGHT, SMILE"
    elif captured == 2:
        instruction = "Photo 3: Turn head SLIGHTLY LEFT"
    elif captured == 3:
        instruction = "Photo 4: Turn head SLIGHTLY RIGHT"
    elif captured == 4:
        instruction = "Photo 5: Look UP slightly"
    elif captured == 5:
        instruction = "Photo 6: Look DOWN slightly"
    elif captured == 6:
        instruction = "Photo 7: SERIOUS expression, front"
    elif captured == 7:
        instruction = "Photo 8: SMILE BIG, front"
    elif captured == 8:
        instruction = "Photo 9: Turn LEFT, smile"
    else:
        instruction = "Photo 10: Turn RIGHT, smile"

    cv2.putText(display, instruction, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Photo {captured + 1}/{SAMPLES_PER_PERSON} - Press SPACE", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw face rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Registration - SPACE to capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("Cancelled")
        cam.release()
        cv2.destroyAllWindows()
        exit()

    elif key == 32:  # SPACE
        if len(faces) == 1:  # Only one face
            # Extract ONLY the face region with minimal margin
            (x, y, w, h) = faces[0]

            # Small margin to avoid cutting face, but minimize background
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            # Extract and resize face to standard size
            face_only = frame[y1:y2, x1:x2]
            face_resized = cv2.resize(face_only, (224, 224))

            # Save the cropped face (NOT the whole frame!)
            filename = os.path.join(person_folder, f"sample_{captured + 1}.jpg")
            cv2.imwrite(filename, face_resized)

            print(f"  ✓ Captured photo {captured + 1}/{SAMPLES_PER_PERSON} (cropped to face only)")

            # Generate embedding using DeepFace
            try:
                print(f"    Generating deep learning embedding...")
                embedding = DeepFace.represent(
                    img_path=filename,
                    model_name="VGG-Face",  # Best for accuracy
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(embedding)
                print(f"    ✓ Embedding generated (4096 dimensions)")

            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

            captured += 1

            # Flash effect
            cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]),
                         (255, 255, 255), 30)
            cv2.imshow("Registration - SPACE to capture", display)
            cv2.waitKey(200)
        else:
            print("  ✗ Error: Found multiple faces or no face. Show only ONE face.")

cam.release()
cv2.destroyAllWindows()

if len(embeddings) == SAMPLES_PER_PERSON:
    # Save embeddings
    embedding_file = os.path.join(person_folder, "embeddings.pkl")
    with open(embedding_file, 'wb') as f:
        pickle.dump({
            'name': person_name,
            'id': person_id,
            'embeddings': embeddings,
            'average_embedding': np.mean(embeddings, axis=0)
        }, f)

    print(f"\n✓ Registration complete!")
    print(f"  Name: {person_name}")
    print(f"  ID: {person_id}")
    print(f"  Embeddings: {len(embeddings)}")
    print(f"  Quality: ULTRA-HIGH (Deep Learning)")
    print(f"\nNext: python ultra_train.py")
else:
    print(f"\n✗ Registration incomplete ({len(embeddings)}/{SAMPLES_PER_PERSON})")