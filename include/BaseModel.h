#ifndef BASEMODEL_H
#define BASEMODEL_H

#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <condition_variable> 

template <typename ResultType>
class BaseModel {
public:
    std::mutex mtx_, resultMtx_;                      // 互斥锁
    std::condition_variable cv_;                      // 用于通知读取线程
    bool dataReady_ = false;                          // 标记数据是否更新

    virtual ~BaseModel() = default;

    // 初始化模型
    virtual int init(const std::string& modelPath) = 0;

    // 获取 RKNN context
    virtual rknn_context* get_rknn_context() = 0;

    // 推理函数
    virtual int infer(const cv::Mat& inputData) = 0;

    // 设置回调函数
    void setCallback(std::function<void(ResultType)> cb) {
        callback_ = cb;                                // 设置回调函数
    }

    // 启动模型推理线程
    void run() {
        std::lock_guard<std::mutex> lock(mtx_);
        infer(inputData_);
    }

protected:
    cv::Mat inputData_;                               // 输入数据
    std::string modelPath_;                           // 模型路径
    unsigned char *model_data_;                       // 模型数据
    int channel_, width_, height_;                       // 输入通道、宽度和高度
    int img_width_, img_height_;                        // 图像宽度和高度
    rknn_context ctx_;                                // RKNN上下文
    rknn_input_output_num io_num_;                    // 输入输出数量
    rknn_tensor_attr *input_attrs_;                   // 输入张量属性
    rknn_tensor_attr *output_attrs_;                  // 输出张量属性
    rknn_input inputs_[1];                            // 输入数组
    float nms_threshold_, box_conf_threshold_;        // NMS阈值和置信度阈值
    std::function<void(ResultType)> callback_;        // 存储回调函数
};

#endif // BASEMODEL_H
