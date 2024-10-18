#ifndef FALLDOWNDETECT_H
#define FALLDOWNDETECT_H

#include "BaseModel.h"
#include <opencv2/core.hpp> // 确保包含OpenCV核心模块

// 结构体定义，用于存储检测结果
struct FallDetection {
    int id;                          // 检测到的对象 ID
    float confidence;                // 置信度
    cv::Rect_<float> box;            // 检测框，使用浮点数表示坐标
};

// 跌倒检测结果结构体
struct FallDetResult {
    std::vector<FallDetection> detections;
};

// 类定义，继承自BaseModel，处理跌倒检测
class FallDet : public BaseModel<FallDetResult> {
public:
    // 默认构造函数
    FallDet();

    // 初始化模型
    int init(const std::string& modelPath) override;

    // 获取RKNN上下文
    rknn_context* get_rknn_context() override;

    // 模型推理
    int infer(const cv::Mat& inputData) override;

    // 获取检测结果
    FallDetResult getResult() const;

    // 默认构造函数
    ~FallDet();

private:
    FallDetResult result_;    // 存储检测结果
};

#endif // FALLDOWNDETECT_H
