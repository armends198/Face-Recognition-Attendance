# University Face Attendance System

A face recognition attendance system for tracking student entry and exit times. Built with deep learning to accurately distinguish between similar-looking people.

## Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python numpy deepface tensorflow
```
*Note: First run downloads AI models (~200MB), takes a few minutes*

### 2. Register Students
```bash
python ultra_register.py
```
- Enter student name and ID
- Take 10 photos (follow on-screen instructions)
- Repeat for each student

### 3. Train the System
```bash
python ultra_train.py
```

### 4. Run Attendance
```bash
python ultra_attendance.py
```
Choose mode:
- **Entry (1)**: Track students arriving
- **Exit (2)**: Track students leaving

Results are saved to `attendance_log.csv`

## How It Works

The system uses VGG-Face neural network to extract 4096 facial features from each student. During attendance, it compares detected faces against registered students and logs matches with timestamp and entry/exit type.

**Key features:**
- Distinguishes similar-looking students (siblings, twins)
- Background-independent (works in different rooms)
- 5-minute cooldown between logs
- Separate tracking for entry and exit

## File Structure

```
├── ultra_main.py           # Main menu interface
├── ultra_register.py       # Register new students
├── ultra_train.py          # Train the system
├── ultra_attendance.py     # Run attendance tracking
├── ultra_reset.py          # Delete all data
├── face_database/          # Stored student photos
├── ultra_database.pkl      # Trained model
└── attendance_log.csv      # Attendance records
```

## CSV Output Format

```csv
ID,Name,Date,Time,Type,Similarity,Model
001,John,2025-02-13,08:30:15,ENTRY,95.2%,VGG-Face
001,John,2025-02-13,16:45:30,EXIT,94.8%,VGG-Face
```

## Adjusting Settings

Edit `ultra_attendance.py` if needed:

```python
SIMILARITY_THRESHOLD = 0.35  # Lower = stricter matching
CONFIDENCE_MARGIN = 0.12     # Higher = less mixing of similar faces
COOLDOWN_MINUTES = 5         # Time between repeated logs
```

## Troubleshooting

**Not recognizing students?**
- Make sure you ran `ultra_train.py` after registration
- Try lowering `SIMILARITY_THRESHOLD` to 0.40

**Mixing up similar students?**
- Increase `CONFIDENCE_MARGIN` to 0.15 or 0.20
- Re-register with varied expressions and angles

**Slow recognition?**
- Normal for deep learning (2-3 seconds per face)
- Processing happens every 10 frames to maintain speed

## Requirements

- Python 3.8+
- Webcam
- 2GB free space

## Reset System

To delete all data and start over:
```bash
python ultra_reset.py
```

---

Built with VGG-Face deep learning for accurate face recognition