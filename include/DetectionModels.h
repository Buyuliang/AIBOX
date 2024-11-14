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
#include <DatabaseManager.h>
#include <regex>

#define QUEUE_LENGTH 1000
#define MAX_FRAME_ID (QUEUE_LENGTH + 1)

// 封装退出标志的结构体
struct ExitFlags {
    std::atomic<bool> cap_exit{false};
    std::atomic<bool> infer_exit{false};
    std::atomic<bool> result_exit{false};
};

std::vector<std::string> attribute_labels = {
    "Hat", "Glasses", "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "Boots",
    "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront", "AgeOver60", "Age18-60", "AgeLess18",
    "Female", "Front", "Side", "Back"
};
