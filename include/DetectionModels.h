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

#define QUEUE_LENGTH 1000
#define MAX_FRAME_ID (QUEUE_LENGTH + 1)
