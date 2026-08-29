@echo off
setlocal
cd /d "%~dp0"
"D:\project\step1\env\python.exe" train_yolov8.py %*
endlocal
