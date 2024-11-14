#include "PersonAttribute.h"
#include <rknn_api.h>
#include "FileUtils.h"
#include <thread>

PerAttr::PerAttr() {

}

int PerAttr::init(const std::string& modelPath) {

    std::cout << "Loading FireSmokeDet model " << modelPath << " ..." << std::endl;

    int model_data_size = 0;
    int ret = -1;
    modelPath_ = modelPath;

    // 加载模型数据
    model_data_ = load_model(modelPath_.c_str(), &model_data_size);
    if (!model_data_) {
        std::cerr << "Failed to load model data from: " << modelPath << std::endl;
        return -1;
    }

    // 初始化 RKNN 模型上下文
    ret = rknn_init(&ctx_, model_data_, model_data_size, 0, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_init failed with error code: " << ret << std::endl;
        return -1;
    }

    // // 绑定核心处理器
    // ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_2);
    // if (ret < 0) {
    //     std::cerr << "Failed to set core mask, error code: " << ret << std::endl;
    //     return -1;
    // }

    // 查询 SDK 版本信息
    rknn_sdk_version version;
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        std::cerr << "Failed to query SDK version, error code: " << ret << std::endl;
        return -1;
    }
    std::cout << "SDK version: " << version.api_version << ", driver version: " << version.drv_version << std::endl;

    // 获取模型输入输出参数
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret < 0) {
        std::cerr << "Failed to query model I/O number, error code: " << ret << std::endl;
        return -1;
    }
    std::cout << "Model input num: " << io_num_.n_input << ", output num: " << io_num_.n_output << std::endl;

    // 分配并设置输入参数
    input_attrs_ = static_cast<rknn_tensor_attr*>(calloc(io_num_.n_input, sizeof(rknn_tensor_attr)));
    if (!input_attrs_) {
        std::cerr << "Failed to allocate memory for input attributes." << std::endl;
        return -1;
    }

    for (int i = 0; i < io_num_.n_input; i++) {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            std::cerr << "Failed to query input attribute for index " << i << ", error code: " << ret << std::endl;
            return -1;
        }
        // dump_tensor_attr(&input_attrs_[i]);
    }

    // 分配并设置输出参数
    output_attrs_ = static_cast<rknn_tensor_attr*>(calloc(io_num_.n_output, sizeof(rknn_tensor_attr)));
    if (!output_attrs_) {
        std::cerr << "Failed to allocate memory for output attributes." << std::endl;
        return -1;
    }

    for (int i = 0; i < io_num_.n_output; i++) {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            std::cerr << "Failed to query output attribute for index " << i << ", error code: " << ret << std::endl;
            return -1;
        }
        // dump_tensor_attr(&output_attrs_[i]);
    }

    // 确定输入格式及维度
    if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
        std::cout << "Model input format: NCHW" << std::endl;
        channel_ = input_attrs_[0].dims[1];
        height_ = input_attrs_[0].dims[2];
        width_ = input_attrs_[0].dims[3];
    } else {
        std::cout << "Model input format: NHWC" << std::endl;
        height_ = input_attrs_[0].dims[1];
        width_ = input_attrs_[0].dims[2];
        channel_ = input_attrs_[0].dims[3];
    }
    std::cout << "Model input dimensions: height=" << height_ << ", width=" << width_ << ", channel=" << channel_ << std::endl;

    // 设置输入参数
    memset(inputs_, 0, sizeof(inputs_));
    inputs_[0].index = 0;
    inputs_[0].type = RKNN_TENSOR_UINT8;
    inputs_[0].size = width_ * height_ * channel_;
    inputs_[0].fmt = RKNN_TENSOR_NHWC;
    inputs_[0].pass_through = 0;

    return 0;
}

rknn_context* PerAttr::get_rknn_context() {
    // 返回 RKNN context
    return nullptr;
}

int PerAttr::infer(const cv::Mat& inputData) {
    std::lock_guard<std::mutex> lock(mtx_);
    cv::Mat img;

    // 获取并输出图像宽度和高度
    img_width_ = inputData.cols;
    img_height_ = inputData.rows;
    // std::cout << "Image Width: " << img_width_ << ", Height: " << img_height_ << std::endl;

    // 调整图像大小并转换为 RGB 格式
    cv::resize(inputData, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 准备输入数据
    inputs_[0].index = 0;
    inputs_[0].type = RKNN_TENSOR_UINT8;
    inputs_[0].size = img.total() * img.elemSize();
    inputs_[0].fmt = RKNN_TENSOR_NHWC;
    inputs_[0].buf = img.data;
    inputs_[0].pass_through = 0;

    // 设置输入
    int ret = rknn_inputs_set(ctx_, 1, inputs_);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed! ret=" << ret << std::endl;
        return -1;
    }

    // 初始化输出
    rknn_output outputs_[io_num_.n_output];
    memset(outputs_, 0, sizeof(outputs_));
    for (int i = 0; i < io_num_.n_output; i++) {
        outputs_[i].index = i;
        outputs_[i].want_float = 1;
    }

    // 运行推理
    if ((ret = rknn_run(ctx_, nullptr)) != RKNN_SUCC) {
        std::cerr << "rknn_run failed! ret=" << ret << std::endl;
        return -1;
    }

    // 获取输出
    if ((ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs_, nullptr)) != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed! ret=" << ret << std::endl;
        return -1;
    }

    // 处理输出数据
    std::vector<float> outputs_vec;
    if (outputs_[0].buf != nullptr && outputs_[0].size % sizeof(float) == 0) {
        // 假设输出数据是 float 类型
        size_t num_elements = outputs_[0].size / sizeof(float);  // 计算 float 元素个数
        outputs_vec.resize(num_elements);  // 使用 resize 调整 vector 的大小

        // 将数据拷贝到 vector 中
        std::memcpy(outputs_vec.data(), outputs_[0].buf, outputs_[0].size);

        // // 打印 vector 的内容
        // for (const auto& value : outputs_vec) {
        //     std::cout << value << " ";
        // }
        // std::cout << std::endl;
    } else {
        std::cerr << "Invalid data or size mismatch!" << std::endl;
    }

    // 释放输出
    rknn_outputs_release(ctx_, io_num_.n_output, outputs_);
    {
        std::lock_guard<std::mutex> lock(resultMtx_);
        result_.detections.clear();  // 确保目标 vector 是空的
        result_.detections.reserve(outputs_vec.size());  // 预分配空间，避免不必要的内存分配
        if (!outputs_vec.empty()) {
            PerAttrDetection det;
            det.attributes = outputs_vec;
            // 将检测结果添加到结果结构体中
            result_.detections.push_back(det);
        }

        dataReady_ = true;              // 标记数据已更新
        result_.ready_ = true;
        cv_.notify_one();               // 通知等待的线程有新数据
    }
    return 0;
}

PerAttrResult PerAttr::getResult() const {
    return result_;
}

PerAttr::~PerAttr() {
    // 安全销毁 RKNN 上下文
    if (ctx_) {
        rknn_destroy(ctx_);
    }

    // 安全释放模型数据
    if (model_data_ != nullptr) {
        free(model_data_);
        model_data_ = nullptr;
    }

    // 安全释放输入属性
    if (input_attrs_ != nullptr) {
        free(input_attrs_);
        input_attrs_ = nullptr;
    }

    // 安全释放输出属性
    if (output_attrs_ != nullptr) {
        free(output_attrs_);
        output_attrs_ = nullptr;
    }
}