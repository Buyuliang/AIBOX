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
#include <variant>
#include <fstream>
#include <json/json.h>
#include <filesystem>
#include <csignal>

#define QUEUE_LENGTH 1000
#define MAX_FRAME_ID (QUEUE_LENGTH + 1)

// 封装退出标志的结构体
struct ExitFlags {
    std::atomic<bool> cap_exit{false};
    std::atomic<bool> infer_exit{false};
    std::atomic<bool> result_exit{false};
};