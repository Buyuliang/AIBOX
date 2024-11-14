namespace dpool {

ThreadPool::ThreadPool(size_t maxThreads) : quit_(false), currentThreads_(0), maxThreads_(maxThreads) {}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        quit_ = true;
    }
    cv_.notify_all();
    for (auto &elem : threads_) {
        if (elem.second.joinable()) {
            elem.second.join();
        }
    }
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return quit_ || !tasks_.empty(); });
            if (quit_ && tasks_.empty()) {
                --currentThreads_;
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

template <typename Func>
std::future<void> ThreadPool::submit(Func &&task) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();
    tasks_.emplace([promise, task]() {
        task();  // 执行任务
        promise->set_value();  // 标记任务完成
    });

    if (currentThreads_ < maxThreads_) {
        std::thread t(&ThreadPool::worker, this);
        threads_[t.get_id()] = std::move(t);
        ++currentThreads_;
    } else {
        cv_.notify_one();
    }
    return future;
}

}  // namespace dpool
