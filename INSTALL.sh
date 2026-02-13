#!/bin/bash

echo "===================================================================="
echo "ULTRA-ACCURATE FACE ATTENDANCE SYSTEM - INSTALLATION"
echo "===================================================================="
echo ""
echo "Installing required libraries..."
echo "This will take 2-5 minutes and download ~200MB"
echo ""
read -p "Press Enter to continue..."

pip3 install opencv-python numpy pillow deepface tensorflow

echo ""
echo "===================================================================="
echo "INSTALLATION COMPLETE!"
echo "===================================================================="
echo ""
echo "Next steps:"
echo "  1. python3 ultra_register.py  (register people)"
echo "  2. python3 ultra_train.py     (train system)"
echo "  3. python3 ultra_attendance.py (check attendance)"
echo ""
echo "===================================================================="
