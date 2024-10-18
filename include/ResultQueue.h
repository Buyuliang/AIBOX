#ifndef RESULTQUEUE_H
#define RESULTQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename ResultType>
struct ResultWithFrameID {
    ResultType result;
    uint64_t frameID;

    ResultWithFrameID(const ResultType& r, uint64_t id) : result(r), frameID(id) {}
    ResultWithFrameID(ResultType&& r, uint64_t id) : result(std::move(r)), frameID(id) {}
};

template <typename ResultType>
class ResultQueue {
public:
    // 将结果推入队列
    void push(const ResultType& result, uint64_t frameID) {
        std::lock_guard<std::mutex> lock(mtx);
        results.emplace(result, frameID);
        cv.notify_one();  // 通知等待的线程
    }

    void push(ResultType&& result, uint64_t frameID) {
        std::lock_guard<std::mutex> lock(mtx);
        results.emplace(std::forward<ResultType>(result), frameID);
        cv.notify_one();  // 通知等待的线程
    }

    // 从队列中弹出结果，返回 std::optional<ResultWithFrameID<ResultType>> 以处理空队列情况
    std::optional<ResultWithFrameID<ResultType>> pop() {
        std::lock_guard<std::mutex> lock(mtx);
        if (results.empty()) {
            return std::nullopt;  // 返回无效状态
        }
        ResultWithFrameID<ResultType> result = std::move(results.front());
        results.pop();
        return result;
    }

    // 检查队列是否为空
    bool isEmpty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return results.empty();
    }

    // 等待并弹出结果
    ResultWithFrameID<ResultType> waitAndFetchResult() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !results.empty(); }); // 等待直到队列不为空
        ResultWithFrameID<ResultType> result = std::move(results.front());
        results.pop();
        return result;
    }

private:
    mutable std::mutex mtx;                   // 互斥锁
    std::condition_variable cv;                // 条件变量
    std::queue<ResultWithFrameID<ResultType>> results; // 存储结果的队列
};

#endif // RESULTQUEUE_H
