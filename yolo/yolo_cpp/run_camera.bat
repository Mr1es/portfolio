@echo off
REM ============================================================
REM Run YOLO realtime detection on webcam (ONNX Runtime backend)
REM   run_camera.bat
REM Press 'q' in the window to quit.
REM ============================================================
set PATH=D:\opencv\opencv\build\x64\vc16\bin;%PATH%

yolo_ort.exe --model models\yolo26n.onnx --source 0 --show --conf 0.3
