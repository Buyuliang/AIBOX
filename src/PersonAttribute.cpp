#include "PersonAttribute.h"
#include <rknn_api.h>
#include <thread>

int PerAttr::init(const std::string& modelPath) {
    // 初始化模型加载逻辑
    return 0;
}

rknn_context* PerAttr::get_rknn_context() {
    // 返回 RKNN context
    return nullptr;
}

int PerAttr::infer(const cv::Mat& inputData) {
    // 人体属性推理逻辑，填充 result 结果
    std::this_thread::sleep_for(std::chrono::seconds(1));  // 休眠 1 秒
    std::cout << "&" << std::flush;
    return 0;
}

PerAttrResult PerAttr::getResult() const {
    return result_;
}
