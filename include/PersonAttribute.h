#ifndef PERSONATTRIBUTE_H
#define PERSONATTRIBUTE_H

#include "BaseModel.h"
#include <vector>
#include <opencv2/core.hpp> // 确保包含OpenCV核心模块

// 结构体定义，用于存储检测结果
struct PerAttrDetection {
    int id;                          // 检测到的对象 ID
    std::vector<float> attributes;   // 属性数据，使用浮点数表示概率
};

// 跌倒检测结果结构体
struct PerAttrResult {
    std::vector<PerAttrDetection> detections;
    bool ready_; // 标志检测结果是否准备好

    // 默认构造函数
    PerAttrResult() : ready_(false) {} // 初始化 ready 为 false

    // 复制构造函数
    PerAttrResult(const PerAttrResult& other)
        : detections(other.detections), ready_(other.ready_) {}

    // 赋值操作符
    PerAttrResult& operator=(const PerAttrResult& other) {
        if (this != &other) { // 防止自我赋值
            detections = other.detections; // 深拷贝 detections
            ready_ = other.ready_; // 复制 ready
        }
        return *this;
    }
};

// 类定义，继承自BaseModel
class PerAttr : public BaseModel<PerAttrResult> {
public:
    // 默认构造函数
    PerAttr();

    // 初始化模型
    int init(const std::string& modelPath) override;

    // 获取RKNN上下文
    rknn_context* get_rknn_context() override;

    // 模型推理
    int infer(const cv::Mat& inputData) override;

    // 获取检测结果
    PerAttrResult getResult() const;

    // 默认构造函数
    ~PerAttr();

private:
    PerAttrResult result_;    // 存储检测结果
};

#endif // PERSONATTRIBUTE_H
