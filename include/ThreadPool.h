#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <functional>
#include <future>

namespace dpool {

class ThreadPool {
public:
    explicit ThreadPool(size_t maxThreads);
    ~ThreadPool();

    template <typename Func>
    std::future<void>  submit(Func &&task);

private:
    void worker();

    std::mutex mutex_;
    std::condition_variable cv_;
    bool quit_;
    size_t currentThreads_;
    size_t maxThreads_;
    std::queue<std::function<void()>> tasks_;
    std::unordered_map<std::thread::id, std::thread> threads_;
};

}  // namespace dpool

#include "ThreadPool.inl"

#endif // THREADPOOL_H
