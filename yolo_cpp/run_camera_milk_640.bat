@echo off
REM ============================================================
REM Realtime milk-box detection on webcam - 640 version (faster)
REM Model : milk_640.onnx  (RYmilk / MNmilk, imgsz 640)
REM Expect ~6 FPS on CPU, vs ~2 FPS for the 1280 version.
REM Press 'q' in the window to quit.
REM ============================================================
set PATH=D:\opencv\opencv\build\x64\vc16\bin;%PATH%

yolo_ort.exe --model models\milk_640.onnx --source 0 --show --conf 0.3
