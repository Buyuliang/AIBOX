#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include "ThreadPool.h"
#include <queue>
#include <memory>
#include <future>
#include "MutexQueue.h"

// using DetectionResult = std::variant<PerDetResult, PerAttrResult, FallDetResult, FireSmokeDetResult>;

// rknnModel模型类, inputType模型输入类型
template <typename rknnModel, typename inputType, typename resultType>
class rknnPool {
private:
    int threadNum_;                                      // 线程数量
    std::string modelPath_;                              // 模型路径
    long long id_;                                       // 任务ID计数器
    std::mutex idMtx_, queueMtx_;                        // 线程安全的互斥锁
    std::unique_ptr<dpool::ThreadPool> pool_;            // 线程池实例
    std::queue<std::future<void>> futs_;                 // 存储推理结果的future队列
    std::vector<std::shared_ptr<rknnModel>> models_;     // 模型实例集合
    MutexQueue& resultQueue_;          // 结果队列引用

protected:
    // 获取模型ID，用于调度模型
    int getModelId();

public:
    // 构造函数：初始化模型路径和线程数
    rknnPool(const std::string& modelPath, int threadNum, MutexQueue& resultQueue);
    // rknnPool(const std::string& modelPath, int threadNum, ResultQueue<DetectionResult>& resultQueue);

    // 初始化每个模型实例
    int init();

    // 模型推理：将输入数据放入线程池进行处理
    int put(inputType inputData, uint64_t frameID);

    // 获取推理结果：从 future 队列中获取推理结果
    // int get(DetectionResult& outputData);

    // 析构函数：释放资源
    ~rknnPool();
};

#include "RknnPool.inl"

#endif
