// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// 基于官方 ultralytics/examples/cpp/ONNXRuntime/main.cpp 改造：
//   官方原版只推理单张图片，这里增加了摄像头 / 视频实时循环，并加上 FPS 统计。
//   其余推理流程（letterbox、解码、NMS）全部复用官方的 inference.cpp 与 common/ 头文件。
//
// 用法：
//   摄像头实时：  yolo_onnxruntime.exe --model yolo26n.onnx --source 0 --show
//   视频文件：    yolo_onnxruntime.exe --model yolo26n.onnx --source test.mp4 --show
//   单张图片：    yolo_onnxruntime.exe --model yolo26n.onnx --source bus.jpg --show
//   可选参数：    --conf 0.25  --iou 0.45  --cuda（需 CUDA 版 ONNX Runtime）

#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference.h"
#include "yolo_cli.hpp"
#include "yolo_draw.hpp"
#include "yolo_render.hpp"
#include "yolo_show.hpp"

namespace {

// 判断 --source 是设备序号（摄像头）还是文件路径
bool IsDeviceIndex(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

// 在画面左上角画 FPS 和目标数
void DrawHud(cv::Mat& image, double fps, size_t objects) {
    char text[64];
    snprintf(text, sizeof(text), "FPS: %.1f  |  objects: %zu", fps, objects);
    cv::putText(image, text, cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, text, cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(60, 220, 60), 2, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
    yolo::Config config;
    config.model_path = yolo::ArgValue(argc, argv, "--model", "yolo26n.onnx");
    config.conf = std::stof(yolo::ArgValue(argc, argv, "--conf", "0.25"));
    config.iou = std::stof(yolo::ArgValue(argc, argv, "--iou", "0.45"));
    config.cuda = yolo::HasFlag(argc, argv, "--cuda");
    const std::string source = yolo::ArgValue(argc, argv, "--source", "0");
    const std::string output = yolo::ArgValue(argc, argv, "--out", "result.jpg");
    const bool show = yolo::ShowRequested(argc, argv);

    yolo::Predictor predictor(config);
    const std::vector<std::string>& names = predictor.names();
    std::cout << "Model: " << config.model_path << " | task: " << yolo::TaskName(predictor.task())
              << " | classes: " << names.size() << " | conf=" << config.conf << std::endl;

    // ---------------- 摄像头 / 视频流实时模式 ----------------
    if (IsDeviceIndex(source)) {
        cv::VideoCapture cap(std::stoi(source));
        if (!cap.isOpened()) {
            std::cerr << "ERROR: cannot open camera " << source << std::endl;
            return 1;
        }
        std::cout << "Camera " << source << " opened. Press 'q' to quit." << std::endl;

        cv::Mat frame, semantic;
        double fps = 0.0;
        int frames = 0;

        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "ERROR: failed to grab frame" << std::endl;
                break;
            }
            const int64 t0 = cv::getTickCount();
            std::vector<yolo::Result> results = predictor.predict(frame, semantic);
            const double dt = (cv::getTickCount() - t0) / cv::getTickFrequency();
            const double inst = dt > 0 ? 1.0 / dt : 0.0;
            fps = (frames == 0) ? inst : (fps * 0.9 + inst * 0.1);
            ++frames;

            if (frames % 30 == 0) {
                std::cout << "frame " << frames << " | fps " << fps << " | objects " << results.size()
                          << std::endl;
            }

            if (show) {
                cv::Mat canvas = frame.clone();
                for (const yolo::Result& r : results) {
                    yolo::DrawBox(canvas, r.box, yolo::Label(yolo::NameOf(names, r.class_id), r.confidence),
                                  r.class_id);
                }
                DrawHud(canvas, fps, results.size());
                cv::imshow("YOLO realtime (press q to quit)", canvas);
                if (cv::waitKey(1) == 'q') break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
        std::cout << "Done. frames=" << frames << " avg_fps=" << fps << std::endl;
        return 0;
    }

    // ---------------- 单张图片模式（沿用官方逻辑） ----------------
    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        std::cerr << "ERROR: could not read image '" << source << "'" << std::endl;
        return 1;
    }

    cv::Mat semantic;
    std::vector<yolo::Result> results = predictor.predict(image, semantic);

    cv::Mat canvas = image.clone();
    yolo::RenderAndPrint(canvas, predictor.task(), results, names, semantic);

    cv::imwrite(output, canvas);
    std::cout << "Result image written to " << output << std::endl;
    yolo::Show("YOLO", canvas, show);
    return 0;
}
