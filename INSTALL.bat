@echo off
echo ====================================================================
echo ULTRA-ACCURATE FACE ATTENDANCE SYSTEM - INSTALLATION
echo ====================================================================
echo.
echo Installing required libraries...
echo This will take 2-5 minutes and download ~200MB
echo.
pause

pip install opencv-python numpy pillow deepface tensorflow

echo.
echo ====================================================================
echo INSTALLATION COMPLETE!
echo ====================================================================
echo.
echo Next steps:
echo   1. python ultra_register.py  (register people)
echo   2. python ultra_train.py     (train system)
echo   3. python ultra_attendance.py (check attendance)
echo.
echo ====================================================================
pause
