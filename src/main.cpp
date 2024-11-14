#include "RknnPool.h"
#include "DetectionModels.h"
#include <variant>
#include <limits>
#include "MutexQueue.h"
#include <opencv2/opencv.hpp> // 使用 OpenCV 处理图像

ImageDataQueue g_imageData(QUEUE_LENGTH);
MutexQueue g_frameData(QUEUE_LENGTH);

static std::string rtsp_url;
ExitFlags g_flags;
std::atomic<bool> perattr_flags{false};
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

int threadNum = 1;
std::atomic<uint64_t> frameID{0}; // 帧ID

std::string getCurrentTimeStr() {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
    char buffer[80];
    std::strftime(buffer, 80, "%Y%m%d%H%M%S", now);
    return std::string(buffer);
}

void exit_frees() {
    g_flags.cap_exit = true;
    g_flags.infer_exit = true;
    g_flags.result_exit = true;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    g_imageData.clear();
    g_frameData.clear();
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Exiting program..." << std::endl;
    exit_frees();
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

std::string extract_ip(const std::string& rtsp_url) {
    std::regex ip_regex(R"((\d{1,3}\.){3}\d{1,3})"); // 匹配IPv4地址的正则表达式
    std::smatch match;
    if (std::regex_search(rtsp_url, match, ip_regex) && !match.empty()) {
        return match.str();
    }
    return "0.0.0.0"; // 默认值，如果没有找到IP
}

// 插入 RTSP 相关信息到数据库
void insertRTSPLog(sqlite3* db, const std::string& timestamp, const std::string& ip_address, const std::string& rtsp_url) {
    const char* sql = "INSERT INTO rtsp_logs (timestamp, ip_address, rtsp_url) VALUES (?1, ?2, ?3);";
    sqlite3_stmt* stmt;

    // Prepare SQL statement
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, 0) != SQLITE_OK) {
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Bind parameters
    sqlite3_bind_text(stmt, 1, timestamp.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, ip_address.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, rtsp_url.c_str(), -1, SQLITE_STATIC);

    // Execute the statement
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::cerr << "Error executing statement: " << sqlite3_errmsg(db) << std::endl;
    } else {
        std::cout << "Log inserted successfully!" << std::endl;
    }

    // Finalize the statement
    sqlite3_finalize(stmt);
}

void captureFrames(ExitFlags& flags, const std::string& frameSrc) {
    // cv::VideoCapture capture("rtsp://192.168.202.217:554/stream1");
    // cv::VideoCapture capture("/dev/video1");
    cv::VideoCapture capture(frameSrc);
    cv::Mat inputImage;
    std::filesystem::create_directories("output/src");
    // 初始化帧数和时间
    uint64_t frameCount = 0;
    double fps = 0.0;
    auto lastTime = std::chrono::high_resolution_clock::now();  // 记录开始时间

    while (!flags.cap_exit) {
        auto currentFrameTime = std::chrono::high_resolution_clock::now();  // 每帧的时间

        if (!(capture.isOpened() && capture.read(inputImage))) {
            std::cout << "capture exit\n" << std::flush;
            exit_frees();
            exit(0);
        }

        if (inputImage.empty()) {
            std::cout << "inputImage empty\n" << std::flush;
            continue;
        }

        std::time_t now = std::time(nullptr);
        std::tm* local_time = std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(local_time, "%Y-%m-%d %H:%M:%S");
        std::string timestamp = oss.str();

        // 将数据插入数据库
        // insertRTSPLog(db, timestamp, extract_ip(), frameSrc);

        // 计算经过的时间
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = currentTime - lastTime;

        // 每帧递增帧数
        frameCount++;

        // 如果超过1秒，计算一次FPS并重置帧数
        if (elapsedTime.count() >= 1.0) {
            // 更新帧ID
            uint64_t currentFrameID = frameID.fetch_add(1);

            // 检查帧ID是否溢出
            if (currentFrameID >= MAX_FRAME_ID) {
                frameID.store(0); // 重置帧ID
                std::cout << "reset ID\n" << std::flush;
            }
            fps = frameCount / elapsedTime.count();  // 计算 FPS
            // std::cout << "FPS: " << fps << " frames per second\n" << std::endl;
            
            // 重置时间和帧数
            lastTime = currentTime;
            frameCount = 0;
            // cv::imwrite("output/src/" + timestamp + ".png", inputImage);
            g_imageData.push(currentFrameID, inputImage, timestamp, extract_ip(frameSrc));
        }

        std::cout << "." << std::flush;
        // cv::imshow("Detection.png", inputImage);
        // cv::waitKey(1);
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
        // ImageData* frameData = g_frameData.front();
        if (!imageData) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::cout << "imagedata empty\n" << std::flush;
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

        // g_frameData.push(imageData->frameID, frame);
        g_frameData.push(imageData->frameID, frame, imageData->timestamp, extract_ip(rtsp_url));
        g_imageData.pop();

        // 等待所有线程完成
        perDetThread.join();
        fallDetThread.join();
        fireSmokeDetThread.join();
        resultReadyCond.notify_all();
        std::cout << "*" << std::flush;
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void resultProcessingThread(rknnPool<PerAttr, cv::Mat, PerAttrResult>& perAttrDetPool, ExitFlags& flags) {

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
        resultReadyCond.wait_for(lock, std::chrono::milliseconds(400), [] {
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
                std::vector<std::thread> perAttrThreads; // 存储线程
                for (const auto& detection : frameData->perDetResult.detections) {
                    // 将检测框转换为多边形
                    cv::Rect detectionRect(detection.box.x, detection.box.y, detection.box.width, detection.box.height);
                    std::vector<cv::Point> detectionPolygon = {
                        cv::Point(detectionRect.tl().x, detectionRect.tl().y),  // 左上角
                        cv::Point(detectionRect.br().x, detectionRect.tl().y),  // 右上角
                        cv::Point(detectionRect.br().x, detectionRect.br().y),  // 右下角
                        cv::Point(detectionRect.tl().x, detectionRect.br().y)   // 左下角
                    };
                    cv::Rect box(static_cast<int>(detectionRect.tl().x), static_cast<int>(detectionRect.tl().y),
                                static_cast<int>(detection.box.width), static_cast<int>(detection.box.height));
                    // 判断框是否在原始图像内
                    if (detectionRect.x >= 0 && detectionRect.y >= 0 &&
                        detectionRect.x + detectionRect.width <= perDetImage.cols &&
                        detectionRect.y + detectionRect.height <= perDetImage.rows) {

                        // 确保框在图像内才执行 perAttr 线程
                        cv::Mat image = displayImage(detectionRect).clone();
                        std::thread perAttrDetThread([&perAttrDetPool, image, frameID = frameData->imageData.frameID, ID = detection.id]() {
                            perAttrDetPool.put(image, frameID, ID);
                        });

                        perAttrThreads.push_back(std::move(perAttrDetThread)); // 添加线程
                    }
                    // cv::Mat image = displayImage(box).clone();
                    // std::thread perAttrDetThread([&perAttrDetPool, image, frameID = frameData->imageData.frameID, ID = detection.id]() {
                    //     perAttrDetPool.put(image, frameID, ID);
                    // });
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
                for (auto& t : perAttrThreads) {
                    if (t.joinable()) {
                        t.join(); // 等待每个线程完成
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (frameData->perDetResult.ready_ && !frameData->perAttrResult.detections.empty()) {
                Json::Value perAttrJson;
                for (const auto& detection : frameData->perAttrResult.detections) {
                    Json::Value attr;
                    // 遍历 attributes 数组并映射到 JSON 中
                    attr["id"] = detection.id;
                    for (size_t i = 0; i < detection.attributes.size(); ++i) {
                        if (i < attribute_labels.size()) {
                            attr[attribute_labels[i]] = detection.attributes[i] > 0.5;
                        }
                    }
                    perAttrJson.append(attr);
                }
                root["perAttrDetections"] = perAttrJson;
                // cv::imwrite("output/perdet/" + timeStr + ".png", perDetImage);
                std::ofstream perdetFile("output/perdet/" + timeStr + ".json");
                perdetFile << root["perAttrDetections"].toStyledString();  // 写入 JSON 数据
                // std::cout << root["perAttrDetections"].toStyledString() << std::flush;
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
            DatabaseManager dbManager("data.db");
            dbManager.insertLog(frameData->imageData.timestamp, frameData->imageData.ip, rtsp_url, root.toStyledString());
        }
        // 休眠
        // std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // 移除已处理的帧数据
        g_frameData.pop();
        std::cout << "_" << std::flush;
    }
}

int main(int argc, char* argv[]) {

    const std::string modelPath = std::filesystem::path(argv[0]).parent_path().string() + "/model/";
    std::cout << "Current working directory: " << modelPath << std::endl;
    const std::string modelPathPerDet = modelPath + "perdet.rknn";
    const std::string modelPathPerAttr = modelPath + "perattr.rknn";
    const std::string modelPathFallDet = modelPath + "falldet.rknn";
    const std::string modelPathFireSmokeDet = modelPath + "firesmoke.rknn";

    // 检查命令行参数数量
    if (argc < 2) {
        std::cerr << "Error: Not enough arguments provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image_source>" << std::endl;
        return 1;
    }
    //    // 创建数据库实例
    // DatabaseManager dbManager("date.db");

    signal(SIGINT, signalHandler);
    // 从命令行参数获取图像源
    std::string frameSrc = argv[1];
    rtsp_url = argv[1];
    std::cout << "Using image source: " << frameSrc << std::endl;

    // 初始化模型池
    rknnPool<PerDet, cv::Mat, PerDetResult> perDetPool(modelPathPerDet, threadNum, g_frameData);
    perDetPool.init();

    rknnPool<PerAttr, cv::Mat, PerAttrResult> perAttrDetPool(modelPathPerAttr, threadNum, g_frameData);
    perAttrDetPool.init();

    rknnPool<FallDet, cv::Mat, FallDetResult> fallDetPool(modelPathFallDet, threadNum, g_frameData);
    fallDetPool.init();

    rknnPool<FireSmokeDet, cv::Mat, FireSmokeDetResult> fireSmokeDetPool(modelPathFireSmokeDet, threadNum, g_frameData);
    fireSmokeDetPool.init();

    std::thread captureThread(captureFrames, std::ref(g_flags), frameSrc);
    std::thread inferThread(inferenceThread, std::ref(perDetPool), std::ref(fallDetPool), std::ref(fireSmokeDetPool), std::ref(g_flags));
    std::thread resultThread(resultProcessingThread, std::ref(perAttrDetPool), std::ref(g_flags));

    captureThread.join();
    inferThread.join();
    resultThread.join();

    return 0;
}
