"""
RESET SCRIPT - Start Fresh
Deletes all data and starts from zero
"""

import os
import shutil

print("="*70)
print("UNIVERSITY ATTENDANCE SYSTEM - RESET")
print("="*70)
print("\nThis will DELETE:")
print("  - All registered students (face_database/)")
print("  - Trained model (ultra_database.pkl)")
print("  - Attendance records (attendance_log.csv)")
print("  - All cooldown trackers (cooldown_*.pkl)")
print("\n" + "="*70)

response = input("\nType 'YES' to confirm: ").strip()

if response != "YES":
    print("Cancelled.")
    exit()

print("\nDeleting...")

# Delete face database
if os.path.exists('face_database'):
    shutil.rmtree('face_database')
    print("✓ Deleted face_database/")

# Delete trained model
if os.path.exists('ultra_database.pkl'):
    os.remove('ultra_database.pkl')
    print("✓ Deleted ultra_database.pkl")

# Delete attendance
if os.path.exists('attendance_log.csv'):
    os.remove('attendance_log.csv')
    print("✓ Deleted attendance_log.csv")

# Also delete old attendance file if it exists
if os.path.exists('ultra_attendance.csv'):
    os.remove('ultra_attendance.csv')
    print("✓ Deleted ultra_attendance.csv (old)")

# Delete all cooldown tracker files
import glob
for cooldown_file in glob.glob('cooldown_*.pkl'):
    os.remove(cooldown_file)
    print(f"✓ Deleted {cooldown_file}")

# Delete old cooldown file if exists
if os.path.exists('cooldown_tracker.pkl'):
    os.remove('cooldown_tracker.pkl')
    print("✓ Deleted cooldown_tracker.pkl (old)")

# Recreate folders
os.makedirs('face_database', exist_ok=True)

print("\n" + "="*70)
print("✅ RESET COMPLETE!")
print("="*70)
print("\nSystem is clean. Start fresh:")
print("  1. python ultra_register.py")
print("  2. python ultra_train.py")
print("  3. python ultra_attendance.py")
print("="*70)