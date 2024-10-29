#include "RknnPool.h"
#include "DetectionModels.h"
#include <variant>
#include <limits>
#include "MutexQueue.h"
#include <opencv2/opencv.hpp> // 使用 OpenCV 处理图像

ImageDataQueue g_imageData(QUEUE_LENGTH);
MutexQueue g_frameData(QUEUE_LENGTH);
ExitFlags g_flags;
// 添加条件变量和锁
std::condition_variable resultReadyCond;
std::mutex resultMutex;

// 定义多边形框的顶点
std::vector<cv::Point> polygon = {
    cv::Point(350, 50),
    cv::Point(500, 80),
    cv::Point(550, 250),
    cv::Point(400, 300)
};

const std::string modelPath = "/home/mixtile/aiBox/model/";
const std::string modelPathPerDet = modelPath + "perdet.rknn";
const std::string modelPathPerAttr = modelPath + "perattr.rknn";
const std::string modelPathFallDet = modelPath + "falldet.rknn";
const std::string modelPathFireSmokeDet = modelPath + "firesmoke.rknn";
int threadNum = 1;


std::atomic<uint64_t> frameID{0}; // 帧ID

std::string getCurrentTimeStr() {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
    char buffer[80];
    std::strftime(buffer, 80, "%Y%m%d%H%M%S", now);
    return std::string(buffer);
}

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Exiting program..." << std::endl;
    g_flags.cap_exit = true;
    g_flags.infer_exit = true;
    g_flags.result_exit = true;
    exit(signum); 
}

// 判断点是否在多边形框内
bool isPointInPolygon(const std::vector<cv::Point>& polygon, const cv::Point& point) {
    double result = cv::pointPolygonTest(polygon, point, false);
    return result >= 0;
}

// 绘制多边形框的函数
void drawPolygon(cv::Mat& image, const std::vector<cv::Point>& polygon) {
    // 绘制多边形
    const cv::Point* points[1] = { polygon.data() };
    int numberOfPoints[] = { static_cast<int>(polygon.size()) };
    cv::polylines(image, points, numberOfPoints, 1, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

void captureFrames(ExitFlags& flags, const std::string& frameSrc) {
    // cv::VideoCapture capture("rtsp://192.168.202.217:554/stream1");
    // cv::VideoCapture capture("/dev/video1");
    cv::VideoCapture capture(frameSrc);
    cv::Mat inputImage;

// 初始化帧数和时间
uint64_t frameCount = 0;
double fps = 0.0;
auto lastTime = std::chrono::high_resolution_clock::now();  // 记录开始时间

    while (!flags.cap_exit) {
        auto currentFrameTime = std::chrono::high_resolution_clock::now();  // 每帧的时间

        if (!capture.read(inputImage)) {
            flags.cap_exit = true;
            flags.infer_exit = true;
            flags.result_exit = true;
            g_imageData.clear();
            g_frameData.clear();
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

        // 每帧递增帧数
        frameCount++;

        // 计算经过的时间
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = currentTime - lastTime;

        // 如果超过1秒，计算一次FPS并重置帧数
        if (elapsedTime.count() >= 1.0) {
            fps = frameCount / elapsedTime.count();  // 计算 FPS
            std::cout << "FPS: " << fps << " frames per second" << std::endl;
            
            // 重置时间和帧数
            lastTime = currentTime;
            frameCount = 0;
        }

        if (currentFrameID % 6 == 0)
            g_imageData.push(currentFrameID, inputImage.clone()); // 使用 clone()

        cv::imshow("Detection.png", inputImage);
        cv::waitKey(1);
    }
}

void inferenceThread(rknnPool<PerDet, cv::Mat, PerDetResult>& perDetPool,
                     rknnPool<FallDet, cv::Mat, FallDetResult>& fallDetPool,
                     rknnPool<FireSmokeDet, cv::Mat, FireSmokeDetResult>& fireSmokeDetPool, ExitFlags& flags) {
    while (!flags.cap_exit && !flags.infer_exit) {
        if (g_imageData.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ImageData* imageData = g_imageData.front();
        if (!imageData) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue; 
        }
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

void resultProcessingThread(ExitFlags& flags) {

    std::filesystem::create_directories("output/perdet");
    std::filesystem::create_directories("output/falldet");
    std::filesystem::create_directories("output/firesmokedet");
    std::filesystem::create_directories("output/result");
    int count;
    std::string countText;

    while (!flags.result_exit) {
        if (g_frameData.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::unique_lock<std::mutex> lock(resultMutex);
        resultReadyCond.wait_for(lock, std::chrono::milliseconds(150), [] {
            return !g_frameData.empty() &&
                g_frameData.front()->perDetResult.ready_ &&
                g_frameData.front()->fallDetResult.ready_ &&
                g_frameData.front()->fireSmokeDetResult.ready_;
        });

        FrameData* frameData = g_frameData.front();
        if (!frameData) {
            continue;
        }
        cv::Mat origImage = frameData->imageData.frame.clone();
        if (origImage.empty()) {
            continue;
        }
        cv::Mat displayImage = origImage.clone();
        // 获取当前时间字符串用于文件命名
        std::string timeStr = getCurrentTimeStr();

        // 初始化 JSON 对象
        Json::Value root;
        root["frameID"] = static_cast<Json::UInt64>(frameData->imageData.frameID);

        // 处理人检测结果
        if (frameData->perDetResult.ready_) {
            if (!frameData->perDetResult.detections.empty()) {
                Json::Value perDetJson;
                cv::Mat perDetImage = displayImage.clone();
                count = 0;
                for (const auto& detection : frameData->perDetResult.detections) {
                    cv::Scalar color((detection.id * 123) % 256, (detection.id * 456) % 256, (detection.id * 789) % 256);

                    // 绘制矩形框和文本
                    std::string text = std::to_string(detection.id);
                    int baseline = 0;
                    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Point textOrigin(detection.box.x, detection.box.y);
                    cv::Rect textRect(textOrigin, textSize);
                    cv::rectangle(perDetImage, textRect, color, cv::FILLED);
                    putText(perDetImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
                    cv::rectangle(perDetImage, detection.box, color, 1);

                    cv::rectangle(origImage, textRect, color, cv::FILLED);
                    putText(origImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
                    cv::rectangle(origImage, detection.box, color, 1);

                    // 将检测框转换为多边形
                    cv::Rect detectionRect(detection.box.x, detection.box.y, detection.box.width, detection.box.height);
                    std::vector<cv::Point> detectionPolygon = {
                        cv::Point(detectionRect.tl().x, detectionRect.tl().y),  // 左上角
                        cv::Point(detectionRect.br().x, detectionRect.tl().y),  // 右上角
                        cv::Point(detectionRect.br().x, detectionRect.br().y),  // 右下角
                        cv::Point(detectionRect.tl().x, detectionRect.br().y)   // 左下角
                    };

                    // 判断是否与多边形有重叠
                    std::vector<cv::Point> intersectionPolygon;
                    double intersectionArea = cv::intersectConvexConvex(detectionPolygon, polygon, intersectionPolygon);

                    // 如果有重叠则计数
                    if (intersectionArea > 0) {
                        count++;
                    }
                    // 存储 JSON 数据
                    Json::Value det;
                    det["id"] = detection.id;
                    det["x"] = detection.box.x;
                    det["y"] = detection.box.y;
                    det["width"] = detection.box.width;
                    det["height"] = detection.box.height;
                    perDetJson.append(det);
                }
                root["personDetections"] = perDetJson;
                root["RegionCoun"] = count;

                // 保存原始人检测结果图像
                drawPolygon(perDetImage, polygon);
                drawPolygon(origImage, polygon);
                // 在左上角显示 count 变量值
                countText = "Count: " + std::to_string(count);
                cv::putText(perDetImage, countText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

                cv::imwrite("output/perdet/" + timeStr + ".png", perDetImage);
                std::ofstream perdetFile("output/perdet/" + timeStr + ".json");
                perdetFile << root["personDetections"].toStyledString();  // 写入 JSON 数据
                // std::cout << root["personDetections"].toStyledString() << std::flush;
                perdetFile.close();
            }
        }

        // 处理跌倒检测结果
        if (frameData->fallDetResult.ready_) {
            if (!frameData->fallDetResult.detections.empty()) {
                Json::Value fallDetJson;
                cv::Mat fallDetImage = displayImage.clone();
                for (const auto& detection : frameData->fallDetResult.detections) {
                    cv::Scalar color((detection.id * 123) % 256, (detection.id * 456) % 256, (detection.id * 789) % 256);

                    // 绘制矩形框和文本
                    std::string text = std::to_string(detection.id);
                    int baseline = 0;
                    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Point textOrigin(detection.box.x, detection.box.y);
                    cv::Rect textRect(textOrigin, textSize);
                    cv::rectangle(fallDetImage, textRect, color, cv::FILLED);
                    putText(fallDetImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
                    cv::rectangle(fallDetImage, detection.box, color, 1);

                    cv::rectangle(origImage, textRect, color, cv::FILLED);
                    putText(origImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
                    cv::rectangle(origImage, detection.box, color, 1);

                    // 存储 JSON 数据
                    Json::Value det;
                    det["id"] = detection.id;
                    det["x"] = detection.box.x;
                    det["y"] = detection.box.y;
                    det["width"] = detection.box.width;
                    det["height"] = detection.box.height;
                    fallDetJson.append(det);
                }
                root["fallDetections"] = fallDetJson;

                // 保存原始跌倒检测结果图像
                cv::imwrite("output/falldet/" + timeStr + ".png", fallDetImage);
                std::ofstream falldetFile("output/falldet/" + timeStr + ".json");
                falldetFile << root["fallDetections"].toStyledString();  // 写入 JSON 数据
                // std::cout << root["fallDetections"].toStyledString() << std::flush;
                falldetFile.close();
            }
        }

        // 处理火焰烟雾检测结果
        if (frameData->fireSmokeDetResult.ready_) {
            if (!frameData->fireSmokeDetResult.detections.empty()) {
                Json::Value fireSmokeJson;
                cv::Mat fireSmokeDetImage = displayImage.clone();
                for (const auto& detection : frameData->fireSmokeDetResult.detections) {
                    cv::Scalar color((detection.id * 123) % 256, (detection.id * 456) % 256, (detection.id * 789) % 256);

                    // 绘制矩形框和文本
                    std::string text;
                    int baseline = 0;
                    if (detection.id == 0) {
                        text = "fire";
                    } else if (detection.id == 1) {
                        text = "smoke";
                    }

                    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Point textOrigin(detection.box.x, detection.box.y);
                    cv::Rect textRect(textOrigin, textSize);
                    cv::rectangle(fireSmokeDetImage, textRect, color, cv::FILLED);
                    putText(fireSmokeDetImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
                    cv::rectangle(fireSmokeDetImage, detection.box, color, 1);

                    cv::rectangle(origImage, textRect, color, cv::FILLED);
                    putText(origImage, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
                    cv::rectangle(origImage, detection.box, color, 1);

                    // 存储 JSON 数据
                    Json::Value det;
                    det["id"] = text;
                    det["x"] = detection.box.x;
                    det["y"] = detection.box.y;
                    det["width"] = detection.box.width;
                    det["height"] = detection.box.height;
                    fireSmokeJson.append(det);
                }
                root["fireSmokeDetections"] = fireSmokeJson;

                // 保存原始火焰烟雾检测结果图像
                cv::imwrite("output/firesmokedet/" + timeStr + ".png", fireSmokeDetImage);
                std::ofstream firesmokeFile("output/firesmokedet/" + timeStr + ".json");
                firesmokeFile << root["fireSmokeDetections"].toStyledString();  // 写入 JSON 数据
                // std::cout << root["fireSmokeDetections"].toStyledString() << std::flush;
                firesmokeFile.close();
            }
        }
        if ((g_frameData.front()->perDetResult.ready_ ||
                g_frameData.front()->fallDetResult.ready_ ||
                g_frameData.front()->fireSmokeDetResult.ready_) && ( 
                !frameData->perDetResult.detections.empty() ||
                !frameData->fallDetResult.detections.empty() ||
                !frameData->fireSmokeDetResult.detections.empty())) {
            // 保存合成的结果图像
            cv::putText(origImage, countText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            cv::imwrite("output/result/" + timeStr + ".png", origImage);
            std::ofstream resultFile("output/result/" + timeStr + ".json");
            resultFile << root.toStyledString(); 
        }
        // 休眠
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // 移除已处理的帧数据
        g_frameData.pop();
    }
}

int main(int argc, char* argv[]) {
    // 检查命令行参数数量
    if (argc < 2) {
        std::cerr << "Error: Not enough arguments provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image_source>" << std::endl;
        return 1;
    }
    signal(SIGINT, signalHandler);
    // 从命令行参数获取图像源
    std::string frameSrc = argv[1];
    std::cout << "Using image source: " << frameSrc << std::endl;

    // 初始化模型池
    rknnPool<PerDet, cv::Mat, PerDetResult> perDetPool(modelPathPerDet, threadNum, g_frameData);
    perDetPool.init();

    rknnPool<FallDet, cv::Mat, FallDetResult> fallDetPool(modelPathFallDet, threadNum, g_frameData);
    fallDetPool.init();

    rknnPool<FireSmokeDet, cv::Mat, FireSmokeDetResult> fireSmokeDetPool(modelPathFireSmokeDet, threadNum, g_frameData);
    fireSmokeDetPool.init();

    std::thread captureThread(captureFrames, std::ref(g_flags), frameSrc);
    std::thread inferThread(inferenceThread, std::ref(perDetPool), std::ref(fallDetPool), std::ref(fireSmokeDetPool), std::ref(g_flags));
    std::thread resultThread(resultProcessingThread, std::ref(g_flags));

    captureThread.join();
    inferThread.join();
    resultThread.join();

    return 0;
}
