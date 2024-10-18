// include/DetectionModels.h
#pragma once

#include "BaseModel.h"
#include "FalldownDetect.h"
#include "FireSmokeDetect.h"
#include "PersonAttribute.h"
#include "PersonDetect.h"
#include "RknnPool.h"
#include "ThreadPool.h"
#include "ResultQueue.h"
#include "MutexQueue.h"
#include <variant>

struct FrameData {
    int frame_id;       // 帧ID
    cv::Mat image;      // 图像数据，使用 OpenCV 的 Mat 类型

    FrameData(int id, const cv::Mat& img) : frame_id(id), image(img.clone()) {} // 构造函数
};
