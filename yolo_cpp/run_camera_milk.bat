@echo off
REM ============================================================
REM Realtime milk-box detection on webcam (ONNX Runtime backend)
REM Model : milk_best.onnx  (RYmilk / MNmilk, trained at imgsz 1280)
REM Press 'q' in the window to quit.
REM ============================================================
set PATH=D:\opencv\opencv\build\x64\vc16\bin;%PATH%

yolo_ort.exe --model models\milk_best.onnx --source 0 --show --conf 0.3
