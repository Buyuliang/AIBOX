#ifndef PERSONATTRIBUTE_H
#define PERSONATTRIBUTE_H

#include "BaseModel.h"
#include <vector>
#include <string>         // 确保包含string头文件
#include <opencv2/core.hpp> // 确保包含OpenCV核心模块

// 结构体定义，用于存储人物属性检测结果
struct PerAttrResult {
    cv::Rect faceBox;         // 人脸框
    int age;                  // 年龄
    std::string emotion;      // 情绪
};

// 人物属性检测类，继承自BaseModel
class PerAttr : public BaseModel<PerAttrResult> {
public:
    // 初始化模型
    int init(const std::string& modelPath) override;

    // 获取RKNN上下文
    rknn_context* get_rknn_context() override;

    // 模型推理
    int infer(const cv::Mat& inputData) override;

    // 获取检测结果
    PerAttrResult getResult() const;

private:
    PerAttrResult result_; // 存储检测结果
};

#endif // PERSONATTRIBUTE_H
