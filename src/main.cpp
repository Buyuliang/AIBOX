#include "RknnPool.h"
#include "DetectionModels.h"
#include <variant>
#include <limits>
#include "MutexQueue.h"
#include <opencv2/opencv.hpp> // 使用 OpenCV 处理图像

ImageDataQueue g_imageData(QUEUE_LENGTH);
MutexQueue g_frameData(QUEUE_LENGTH);
// 添加条件变量和锁
std::condition_variable resultReadyCond;
std::mutex resultMutex;

const std::string modelPath = "/home/mixtile/aiBox/model/";
const std::string modelPathPerDet = modelPath + "perdet.rknn";
const std::string modelPathPerAttr = modelPath + "perattr.rknn";
const std::string modelPathFallDet = modelPath + "falldet.rknn";
const std::string modelPathFireSmokeDet = modelPath + "firesmoke.rknn";
int threadNum = 1;

// 模拟输入数据
// cv::Mat inputImage = cv::imread("/home/mixtile/aiBox/sources/test.jpg");
// cv::Mat inputImage = cv::imread("/home/mixtile/aiBox/sources/fire.png");
// cv::Mat inputImage = cv::imread("/home/mixtile/aiBox/sources/fall.png");


std::atomic<uint64_t> frameID{0}; // 帧ID

void captureFrames(bool& _exit, bool& infer_exit, bool& result_exit) {
    // cv::VideoCapture capture("/home/mixtile/aiBox/sources/fall.png");
    cv::VideoCapture capture("/home/mixtile/tools/test/01_Python/09_sort_rknn/test.mp4");
    cv::Mat inputImage;

    while (!_exit) {
        if (!capture.read(inputImage)) {
            g_imageData.clear();
            _exit = true;
            std::cout << "capture break" << std::flush;
            break;
        }

        if (inputImage.empty()) {
            std::cout << "inputImage empty" << std::flush;
            continue;
        }

        // 更新帧ID
        uint64_t currentFrameID = frameID.fetch_add(1);

        // 检查帧ID是否溢出
        if (currentFrameID >= MAX_FRAME_ID) {
            frameID.store(0); // 重置帧ID
            std::cout << "reset ID" << std::flush;
        }

        g_imageData.push(currentFrameID, inputImage.clone()); // 使用 clone()

        // cv::imshow("Detection.png", inputImage);
        // cv::waitKey(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // std::cout << "-" << std::flush;
    }
}

void inferenceThread(rknnPool<PerDet, cv::Mat, PerDetResult>& perDetPool,
                     rknnPool<FallDet, cv::Mat, FallDetResult>& fallDetPool,
                     rknnPool<FireSmokeDet, cv::Mat, FireSmokeDetResult>& fireSmokeDetPool, bool& _exit) {
    while (!_exit) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (g_imageData.empty()) {
            continue;
        }

        ImageData* imageData = g_imageData.front();
        cv::Mat frame = imageData->frame.clone();

        // 创建线程来并行执行 put 操作
        std::thread perDetThread([&perDetPool, frame, id = imageData->frameID]() {
            perDetPool.put(frame, id);
        });
        
        std::thread fallDetThread([&fallDetPool, frame, id = imageData->frameID]() {
            fallDetPool.put(frame, id);
        });

        std::thread fireSmokeDetThread([&fireSmokeDetPool, frame, id = imageData->frameID]() {
            fireSmokeDetPool.put(frame, id);
        });

        g_frameData.push(imageData->frameID, frame);
        g_imageData.pop();

        // 等待所有线程完成
        perDetThread.join();
        fallDetThread.join();
        fireSmokeDetThread.join();
        resultReadyCond.notify_all();
    }
}

void resultProcessingThread(bool& _exit) {
    while (!_exit || !g_frameData.empty()) {
        std::unique_lock<std::mutex> lock(resultMutex);
        resultReadyCond.wait_for(lock, std::chrono::milliseconds(100), [] { return !g_frameData.empty() && 
                                                                                    g_frameData.front()->perDetResult.ready_ &&
                                                                                    g_frameData.front()->fallDetResult.ready_; });
 
        FrameData* frameData = g_frameData.front();
        cv::Mat displayImage = frameData->imageData.frame.clone();
        if (displayImage.empty()) {
            continue;
        }

        if (frameData->perDetResult.ready_) {
            for (const auto& detection : frameData->perDetResult.detections) {
                cv::Scalar color((detection.id * 123) % 256, (detection.id * 456) % 256, (detection.id * 789) % 256);
                // rectangle(displayImage, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);

                // 获取文本大小
                std::string text = std::to_string(detection.id);
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                textSize.width = std::max(textSize.width, static_cast<int>(detection.box.width));

                // 矩形的左上角和右下角坐标
                cv::Point textOrigin(detection.box.x, detection.box.y); // 调整到文本上方
                cv::Rect textRect(textOrigin, textSize);
                cv::rectangle(displayImage, textRect, color, cv::FILLED); // 填充矩形背景

                // 绘制文本，调整位置使其居中
                putText(displayImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255)); // 白色文本
                cv::rectangle(displayImage, detection.box, color, 1);
                // std::cout << "Bounding Box: ("
                //           << detection.box.x << ", "
                //           << detection.box.y << ", "
                //           << detection.box.width << ", "
                //           << detection.box.height << ")\n" << std::flush;
            }
        }

        if (frameData->fallDetResult.ready_) {
            for (const auto& detection : frameData->fallDetResult.detections) {
                cv::Scalar color((detection.id * 123) % 256, (detection.id * 456) % 256, (detection.id * 789) % 256);
                // rectangle(displayImage, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);

                // 获取文本大小
                std::string text = std::to_string(detection.id);
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                textSize.width = std::max(textSize.width, static_cast<int>(detection.box.width));

                // 矩形的左上角和右下角坐标
                cv::Point textOrigin(detection.box.x, detection.box.y); // 调整到文本上方
                cv::Rect textRect(textOrigin, textSize);
                cv::rectangle(displayImage, textRect, color, cv::FILLED); // 填充矩形背景

                // 绘制文本，调整位置使其居中
                putText(displayImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255)); // 白色文本
                cv::rectangle(displayImage, detection.box, color, 1);
                // std::cout << "Bounding Box: ("
                //           << detection.box.x << ", "
                //           << detection.box.y << ", "
                //           << detection.box.width << ", "
                //           << detection.box.height << ")\n" << std::flush;
            }
        }
        cv::imshow("Detection.png", displayImage);
        cv::waitKey(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        // std::cout << "P" << std::flush;
        g_frameData.pop();
    }
}

int main() {

    bool cap_exit = false;
    bool infer_exit = false;
    bool result_exit = false;

    // 初始化模型池
    rknnPool<PerDet, cv::Mat, PerDetResult> perDetPool(modelPathPerDet, threadNum, g_frameData);
    perDetPool.init();

    rknnPool<FallDet, cv::Mat, FallDetResult> fallDetPool(modelPathFallDet, threadNum, g_frameData);
    fallDetPool.init();

    rknnPool<FireSmokeDet, cv::Mat, FireSmokeDetResult> fireSmokeDetPool(modelPathFireSmokeDet, threadNum, g_frameData);
    fireSmokeDetPool.init();

    std::thread captureThread(captureFrames, std::ref(cap_exit), std::ref(infer_exit), std::ref(result_exit));
    std::thread inferThread(inferenceThread, std::ref(perDetPool), std::ref(fallDetPool), std::ref(fireSmokeDetPool), std::ref(infer_exit));
    std::thread resultThread(resultProcessingThread, std::ref(result_exit));
    captureThread.join();

    infer_exit = true;
    result_exit = true;

    inferThread.join();
    resultThread.join();

    return 0;
}
