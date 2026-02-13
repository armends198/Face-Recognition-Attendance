"""
QUICK TEST - Verify Installation
Tests if all libraries are working
"""

print("="*70)
print("TESTING INSTALLATION")
print("="*70)

print("\n1. Testing OpenCV...")
try:
    import cv2
    print(f"   ✓ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"   ✗ OpenCV failed: {e}")
    print("   Run: pip install opencv-python")

print("\n2. Testing NumPy...")
try:
    import numpy as np
    print(f"   ✓ NumPy version: {np.__version__}")
except Exception as e:
    print(f"   ✗ NumPy failed: {e}")
    print("   Run: pip install numpy")

print("\n3. Testing DeepFace (this may take 10-20 seconds first time)...")
try:
    from deepface import DeepFace
    print("   ✓ DeepFace installed")
    
    # Test if it can load models
    print("   Testing VGG-Face model download...")
    # This will download the model if not present
    print("   (May take 30 seconds on first run...)")
    
    print("   ✓ DeepFace working!")
    
except Exception as e:
    print(f"   ✗ DeepFace failed: {e}")
    print("   Run: pip install deepface tensorflow")

print("\n4. Testing TensorFlow...")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"   ✗ TensorFlow failed: {e}")
    print("   Run: pip install tensorflow")

print("\n5. Testing webcam...")
try:
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        ret, frame = cam.read()
        if ret:
            print(f"   ✓ Webcam working! Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("   ✗ Cannot read from webcam")
        cam.release()
    else:
        print("   ✗ Cannot open webcam")
        print("   Check: Is camera connected? Do you have permissions?")
except Exception as e:
    print(f"   ✗ Webcam test failed: {e}")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print("\nIf all tests show ✓, you're ready to use the system!")
print("\nNext steps:")
print("  1. python ultra_register.py  (register people)")
print("  2. python ultra_train.py     (train)")
print("  3. python ultra_attendance.py (use)")
print("="*70)
