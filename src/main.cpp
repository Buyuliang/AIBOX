#include "RknnPool.h"
#include "DetectionModels.h"
#include <variant>
#include <limits>

#include <opencv2/opencv.hpp> // 使用 OpenCV 处理图像

MutexQueue<FrameData> frameDataQueue;
ResultQueue<PerDetResult> perDetResQueue;        // 存储人检测结果
ResultQueue<PerAttrResult> perAttrResQueue;      // 存储人属性检测结果
ResultQueue<FallDetResult> fallDetResQueue;      // 存储跌倒检测结果
ResultQueue<FireSmokeDetResult> fireSmokeDetResQueue; // 存储火焰烟雾检测结果

std::atomic<uint64_t> frameID{0}; // 帧ID

void resetFrameID() {
    std::cerr << "Frame ID has reached its maximum value. Resetting to 0." << std::endl;
    frameID.store(0); // 重置帧ID
}

void processingThread(const cv::Mat& originalImage) {
    uint64_t lastProcessedFrameID = 0;

    while (true) {
        cv::Mat displayImage = originalImage.clone();
        if (displayImage.empty()) {
            continue;
        }

        // 获取当前帧ID
        uint64_t currentFrameID = frameID.load();

        // 检查是否需要处理结果
        if (!perDetResQueue.isEmpty()) {
            auto resultOpt = perDetResQueue.waitAndFetchResult(); // 尝试弹出结果
            // ResultWithFrameID<PerDetResult> result = *resultOpt; // 假设结果总是存在

            // 检查帧ID是否与当前帧匹配
            // if (resultOpt.frameID == currentFrameID) {
                for (const auto& detection : resultOpt.result.detections) {
                    cv::rectangle(displayImage, detection.box, cv::Scalar(0, 255, 0), 2);
                    std::cout << "Bounding Box: ("
                              << detection.box.x << ", "
                              << detection.box.y << ", "
                              << detection.box.width << ", "
                              << detection.box.height << ")" << std::endl;
                }
                lastProcessedFrameID = currentFrameID; // 更新最后处理的帧ID
            // } else {
            //     std::cerr << "Skipped processing frame ID: " << currentFrameID << std::endl;
            // }
        }

        // 显示处理后的图像
        cv::imshow("Detection.png", displayImage);
        cv::waitKey(1);
    }
}

int main() {
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

    // 初始化四个模型池
    rknnPool<PerDet, cv::Mat, PerDetResult> perDetPool(modelPathPerDet, threadNum, perDetResQueue);
    perDetPool.init();

    // rknnPool<PerAttr, cv::Mat> perAttrPool(modelPathPerAttr, threadNum, resultQueue);
    // perAttrPool.init();

    // rknnPool<FallDet, cv::Mat> fallDetPool(modelPathFallDet, threadNum, resultQueue);
    // fallDetPool.init();

    // rknnPool<FireSmokeDet, cv::Mat> fireSmokePool(modelPathFireSmokeDet, threadNum, resultQueue);
    // fireSmokePool.init();

    dpool::ThreadPool pool(threadNum);
    // 打开摄像头
    cv::VideoCapture capture;
    capture.open("/home/mixtile/tools/test/01_Python/09_sort_rknn/test.mp4");
    // capture.open("/dev/video1");
    if (!capture.isOpened()) {
        std::cerr << "Error: Unable to open the camera." << std::endl;
        return -1;
    }
    cv::Mat inputImage;
    std::thread resultThread(processingThread, std::ref(inputImage)); // 启动处理线程

    while (true) {
        if (!capture.read(inputImage)) {
            break;
        }

        // 更新帧ID
        uint64_t currentFrameID = frameID.fetch_add(1); // 递增帧ID

        // 检查帧ID是否溢出
        if (currentFrameID == std::numeric_limits<uint64_t>::max()) {
            resetFrameID(); // 重置帧ID
        }
        FrameData framedata(frameID, inputImage);
        // 将输入数据放入各个模型池
        pool.submit([&, inputImage, currentFrameID]() { perDetPool.put(inputImage.clone(), currentFrameID); });

        if (inputImage.empty()) {
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "-" << std::flush;
    }
    resultThread.join();

    return 0;
}
