#ifndef MUTEX_QUEUE_H
#define MUTEX_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class MutexQueue {
public:
    // 添加元素到队列
    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx); // 锁定队列
        queue.push(std::move(value)); // 移动值到队列中
        cv.notify_one(); // 通知一个等待的线程
    }

    // 从队列中获取元素
    bool pop(T& result) {
        std::lock_guard<std::mutex> lock(mtx); // 锁定队列
        if (queue.empty()) {
            return false; // 如果队列为空，返回 false
        }
        result = std::move(queue.front()); // 移动队首元素到 result
        queue.pop(); // 移除队首元素
        return true; // 返回 true 表示成功
    }

    // 等待并从队列中获取元素
    void wait_and_dequeue(T& result) {
        std::unique_lock<std::mutex> lock(mtx); // 获取锁
        cv.wait(lock, [this] { return !queue.empty(); }); // 等待直到队列不为空
        result = std::move(queue.front()); // 移动队首元素到 result
        queue.pop(); // 移除队首元素
    }

    // 检查队列是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx); // 锁定队列
        return queue.empty(); // 返回队列是否为空
    }

private:
    mutable std::mutex mtx; // 互斥锁
    std::queue<T> queue; // 内部队列
    std::condition_variable cv; // 条件变量
};

#endif // MUTEX_QUEUE_H
