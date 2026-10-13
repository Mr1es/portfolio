@echo off
REM ============================================================
REM Build yolo_ort.exe  (ONNX Runtime backend, recommended)
REM Run this in a normal terminal (not inside the agent sandbox):
REM   build.bat
REM ============================================================
call "D:\vs\VC\Auxiliary\Build\vcvars64.bat"

set ORT=C:\Users\28687\Desktop\X\onnxruntime\onnxruntime-win-x64-1.30.0
set OPENCV=D:\opencv\opencv\build

cl /nologo /std:c++17 /EHsc /O2 /MD ^
  /I"%OPENCV%\include" /I"%ORT%\include" /I"common" ^
  main.cpp inference.cpp /Fe:yolo_ort.exe ^
  /link /LIBPATH:"%OPENCV%\x64\vc16\lib" /LIBPATH:"%ORT%\lib" ^
  opencv_world4130.lib onnxruntime.lib

copy /Y "%ORT%\lib\onnxruntime.dll" .
copy /Y "%ORT%\lib\onnxruntime_providers_shared.dll" .

echo Build done: yolo_ort.exe
