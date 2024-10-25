#include "PersonDetect.h"
#include <rknn_api.h> // RKNN API header
#include <thread>
#include <mutex>
#include "postprocess.h"
#include "preprocess.h"
#include "sort.h"
#include "FileUtils.h"

PerDet::PerDet() {
    nms_threshold_ = 0.45; //NMS_THRESH;      // 默认的NMS阈值
    box_conf_threshold_ = 0.25; //BOX_THRESH; // 默认的置信度阈值
}

int PerDet::init(const std::string& modelPath) {

    std::cout << "Loading PerDet model " << modelPath << " ..." << std::endl;

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

    // 绑定核心处理器
    ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_2);
    if (ret < 0) {
        std::cerr << "Failed to set core mask, error code: " << ret << std::endl;
        return -1;
    }

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

rknn_context* PerDet::get_rknn_context() {
    // 返回 RKNN context
    return nullptr;
}

int PerDet::infer(const cv::Mat& inputData) {
    auto start = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lock(mtx_);
    cv::Mat img;
    int ret;
    result_.ready_ = false;
    cv::cvtColor(inputData, img, cv::COLOR_BGR2RGB);
    img_width_ = img.cols;
    img_height_ = img.rows;
    // std::cout << "inputData size: " << inputData.size() << std::endl;
    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    cv::Size target_size(width_, height_);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
    // 计算缩放比例/Calculate the scaling ratio
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;
    // std::cout << "scale_w:" << scale_w << " scale_h:" << scale_h << std::endl;
    static TrackingSession *sess = CreateSession(2, 3, 0.01);
    // 图像缩放/Image scaling
    if (img_width_ != width_ || img_height_ != height_) {
        // rga
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
        ret = resize_rga(src, dst, img, resized_img, target_size);
        if (ret != 0) {
            fprintf(stderr, "resize with rga error\n");
        }
        /*********
        // opencv
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, resized_img, pads, min_scale, target_size);
        *********/
        inputs_[0].buf = resized_img.data;
    } else {
        inputs_[0].buf = img.data;
    }

    rknn_inputs_set(ctx_, io_num_.n_input, inputs_);

    rknn_output outputs[io_num_.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num_.n_output; i++) {
        outputs[i].want_float = 0;
    }

    // 模型推理/Model inference
    ret = rknn_run(ctx_, NULL);
    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs, NULL);

    // 后处理/Post-processing
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num_.n_output; ++i) {
        out_scales.push_back(output_attrs_[i].scale);
        out_zps.push_back(output_attrs_[i].zp);
    }
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height_, width_,
                 box_conf_threshold_, nms_threshold_, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    char text[256];
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        // sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        // // 打印预测物体的信息/Prints information about the predicted object
        // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        //        det_result->box.right, det_result->box.bottom, det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        // rectangle(inputData, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
        // putText(inputData, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }


    // 生成 SORT 所需的格式
    std::vector<DetectionBox> detections;
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        // if (det_result->prop * 100 >= box_conf_threshold_) { // 置信度过滤 
        if (det_result->prop * 100 >= box_conf_threshold_ && strcmp(det_result->name, "person") == 0) {
            DetectionBox detection;
            detection.box = {det_result->box.left, det_result->box.top, 
                             det_result->box.right - det_result->box.left, 
                             det_result->box.bottom - det_result->box.top};
            detection.score = det_result->prop;
            // detection.class_id = 0; /* 适当的类 ID */;
            detections.push_back(detection);
        }
    }

    // 更新 TrackingSession
    auto trks = sess->Update(detections);
    {
        std::lock_guard<std::mutex> lock(resultMtx_);
        result_.detections.clear();  // 确保目标 vector 是空的
        result_.detections.reserve(trks.size());  // 预分配空间，避免不必要的内存分配

        // 遍历 trackingBoxes，并转换为 Detection
        for (const auto& trackingBox : trks) {
            PerDetection detection;
            detection.id = trackingBox.id;      // 将 track_id 赋值给 Detection 的 id
            detection.box = trackingBox.box;         // 将 rect 赋值给 Detection 的 box

            result_.detections.push_back(detection);          // 将转换后的 Detection 添加到 detections 中
        }
        dataReady_ = true;              // 标记数据已更新
        result_.ready_ = true;
        cv_.notify_one();               // 通知等待的线程有新数据
    }
    // 绘制跟踪框
    // per_num = 0;
    cv::Mat heatmap;
    heatmap = cv::Mat::zeros(inputData.size(), CV_32FC1); 
    if (heatmap_.empty()) {
        heatmap_ = cv::Mat::zeros(inputData.size(), CV_32FC1);  // 创建与原始图像相同大小的热力图
    }
    // std::cout << "inputData.size(): " << inputData.size() << std::endl;
    for (const auto& track : trks) {
        int x1 = track.box.x;
        int y1 = track.box.y;
        int x2 = x1 + track.box.width;
        int y2 = y1 + track.box.height;
        cv::Scalar color((track.id * 123) % 256, (track.id * 456) % 256, (track.id * 789) % 256);
        rectangle(inputData, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);

        // 获取文本大小
        std::string text = std::to_string(track.id);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        textSize.width = std::max(textSize.width, static_cast<int>(track.box.width));

        // 矩形的左上角和右下角坐标
        cv::Point textOrigin(x1, y1); // 调整到文本上方
        // cv::Point textOrigin(x1, y1 - textSize.height - 10); // 调整到文本上方
        cv::Rect textRect(textOrigin, textSize);

        // 绘制矩形背景
        cv::rectangle(inputData, textRect, color, cv::FILLED); // 填充矩形背景

        // 绘制文本，调整位置使其居中
        putText(inputData, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255)); // 白色文本
        // cv::imwrite("test_result.jpg", inputData);

        // int center_y = track.box.y + track.box.height / 2;
        // // 越线检测
        // if (previous_y_positions.size() > track.id) {
        //     if (previous_y_positions[track.id] <= 300 && center_y > 300) {
        //         // if (crossed_up.find(track.id) == crossed_up.end()) {
        //             count_up++;
        //         //     crossed_up.insert(track.id); // 记录已越线 ID
        //         // }
        //     }
        //     // Down crossing
        //     else if (previous_y_positions[track.id] >= 300 && center_y < 300) {
        //         // if (crossed_down.find(track.id) == crossed_down.end()) {
        //             count_down++;
        //         //     crossed_down.insert(track.id); // 记录已越线 ID
        //         // }
        //     }
        // }
        // previous_y_positions[track.id] = center_y;
        // if (isPointInsideRectangle(o_rectangle, cv::Point(track.box.x + track.box.width / 2, track.box.y + track.box.height / 2))) {
        //     per_num++;
        // }

        //处理多对象轨迹
        int mul_x = track.box.x + track.box.width / 2;
        int mul_y = track.box.y + track.box.height / 2;

        // 每个对象使用不同的颜色
        // color = cv::Scalar((track.id * 123) % 256, (track.id * 456) % 256, (track.id * 789) % 256);
        // cv::circle(inputData, cv::Point(mul_x, mul_y), std::min(track.box.width, track.box.height) / 2, color, -1);  // 为每个对象的轨迹绘制点

        // hotmap point
        // cv::Rect_<int> intRect(track.box.x, track.box.y, track.box.width, track.box.height);
        // std::vector<TrackPoint> trackPoints = trackObjects(intRect, track.id);
        // allTrackPoints.insert(allTrackPoints.end(), trackPoints.begin(), trackPoints.end());
        // 统计轨迹点出现次数
        if (track.box.x >= 0 && track.box.x < inputData.cols && track.box.y >= 0 && track.box.y < inputData.rows) {
            for (int y = track.box.y; y < track.box.y + track.box.height; ++y) {
                for (int x = track.box.x; x < track.box.x + track.box.width; ++x) {
                    // 确保 (x, y) 在热力图的有效范围内
                    if (x >= 0 && x < inputData.cols && y >= 0 && y < inputData.rows) {
                        // heatmap.at<float>(y, x) = 100.0;  // 增加热力值
                        heatmap.at<float>(y, x) = std::min(heatmap_.at<float>(y, x) + 1.0f, 255.0f);
                    }
                }
            }
        }
        // for (const auto& point : trackPoints) {
        // int heatpoint_x = track.box.x / scale_w;
        // int heatpoint_y = track.box.y / scale_h;
        // int heatpoint_x = track.box.x;
        // int heatpoint_y = track.box.y;
        // std::cout << "dfdsfasdf-------------------0" << std::endl;
        //     if (heatpoint_x >= 0 && heatpoint_x < inputData.cols && heatpoint_y >= 0 && heatpoint_y < inputData.rows) {
        //         heatmap.at<float>(heatpoint_y, heatpoint_x) += 1.0;  // 每个点出现一次，累加
        //         // heatmap.at<float>(track.box.y, track.box.x) = std::min(heatmap.at<float>(track.box.y, track.box.x) + 1.0f, 255.0f);
        //         cv::circle(inputData, cv::Point(track.box.x, track.box.y), 5, cv::Scalar(1.0f), -1);
        //     }
            // if (track.box.x >= 0 && track.box.x < inputData.cols && track.box.y >= 0 && track.box.y < inputData.rows) {
            //     heatmap.at<float>(track.box.y, track.box.x) += 1.0;  // 每个点出现一次，累加
            //     // heatmap.at<float>(track.box.y, track.box.x) = std::min(heatmap.at<float>(track.box.y, track.box.x) + 1.0f, 255.0f);
            //     cv::circle(inputData, cv::Point(track.box.x, track.box.y), 5, cv::Scalar(1.0f), -1);
            // }
        // }
    }
    // cv::rectangle(ori_img, track.box,  cv::Scalar(0, 255, 0), 2, 8, 0);                                                                                       
    // putText(ori_img, std::to_string(track.id), cv::Point(track.box.x, track.box.y+ 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255, 0));
    // 显示上行和下行人数
    // putText(inputData, "Up through: " + std::to_string(count_up), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    // putText(inputData, "Down through: " + std::to_string(count_down), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    // putText(inputData, "Number of regions: " + std::to_string(per_num), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
     
    ret = rknn_outputs_release(ctx_, io_num_.n_output, outputs);
    // 归一化热力图
    cv::addWeighted(heatmap, 0.01, heatmap_, 0.99, 0.0, heatmap_);
    cv::addWeighted(heatmap, 0.25, heatmap_, 0.75, 0.0, heatmap);
    cv::normalize(heatmap, heatmap, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 将灰度图转换为伪彩色图
    cv::Mat colorHeatmap;
    applyColorMap(heatmap, colorHeatmap, cv::COLORMAP_JET);
    // heatmap *= 0.95;  // 每一帧对所有像素衰减 5%
    // 混合热力图和原始图像，alpha 为原始图像的权重，beta 为热力图的权重
    double alpha = 0.5;  // 原始图像的权重
    double beta = 0.5;   // 热力图的权重
    cv::Mat blended;
    // cv::addWeighted(colorHeatmap, alpha, colorHeatmap, beta, 0.0, blended);
    // cv::circle(colorHeatmap, cv::Point(20, 20), 10, cv::Scalar(1.0f), -1);
    cv::addWeighted(inputData, alpha, colorHeatmap, beta, 0.0, blended);
    auto end = std::chrono::high_resolution_clock::now();

    // 计算推理时间
    std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout << "Inference time: " << duration.count() << " ms" << std::flush;
    // 保存热力图
    // cv::imwrite("test_result.jpg", colorHeatmap);
    // return blended;
    // return inputData;
    return 0;
}

PerDetResult PerDet::getResult() const {
    return result_;
}

PerDet::~PerDet() {
    // 安全清理后处理流程
    deinitPostProcess();

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