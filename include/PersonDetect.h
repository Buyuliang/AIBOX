#ifndef PERSONDETECT_H
#define PERSONDETECT_H

#include "BaseModel.h"
#include <vector>
#include <opencv2/core/core.hpp> // 确保包含OpenCV核心模块

// 检测结果结构体
struct PerDetection {
    int id;                          // 检测到的对象 ID
    cv::Rect_<float> box;            // 检测框，使用浮点数表示坐标
};

// 人物检测结果结构体
struct PerDetResult {
    std::vector<PerDetection> detections; // 存储多个检测结果，每个结果包含 id 和 box
};

// 人物检测类，继承自 BaseModel
class PerDet : public BaseModel<PerDetResult> {
public:
    // 默认构造函数
    PerDet();

    // 初始化模型
    int init(const std::string& modelPath) override;

    // 获取 RKNN 上下文
    rknn_context* get_rknn_context() override;

    // 模型推理
    int infer(const cv::Mat& inputData) override;

    // 获取检测结果
    PerDetResult getResult() const;

    // 默认析构函数
    ~PerDet();

private:
    PerDetResult result_;          // 存储检测结果
    cv::Mat heatmap_;              // 热力图（如果需要）
    float nms_threshold_;          // 非极大值抑制阈值
    float box_conf_threshold_;     // 检测框置信度阈值
};

#endif // PERSONDETECT_H
