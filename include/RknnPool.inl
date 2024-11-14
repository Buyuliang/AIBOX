template <typename rknnModel, typename inputType, typename resultType>
rknnPool<rknnModel, inputType, resultType>::rknnPool(const std::string& modelPath, int threadNum, MutexQueue& resultQueue)
    : modelPath_(modelPath), threadNum_(threadNum), resultQueue_(resultQueue), id_(0){
    pool_ = std::make_unique<dpool::ThreadPool>(threadNum);
}

template <typename rknnModel, typename inputType, typename resultType>
int rknnPool<rknnModel, inputType, resultType>::init() {
    for (int i = 0; i < threadNum_; ++i) {
        auto model = std::make_shared<rknnModel>();
        std::cout << "rknnpool init" << std::endl;
        if (model->init(modelPath_) != 0) {
            std::cerr << "Model initialization failed for thread " << i << std::endl;
            return -1;
        }
        models_.push_back(model);
    }
    return 0;
}

template <typename rknnModel, typename inputType, typename resultType>
int rknnPool<rknnModel, inputType, resultType>::getModelId() {
    std::lock_guard<std::mutex> lock(idMtx_);
    return id_++ % threadNum_;
}

template <typename rknnModel, typename inputType, typename resultType>
int rknnPool<rknnModel, inputType, resultType>::put(inputType inputData, uint64_t frameID, uint64_t ID) {
    std::lock_guard<std::mutex> lock(queueMtx_); // 确保对 futs_ 的安全访问
    futs_.push(pool_->submit([this, inputData, frameID, ID]() {
        // 获取当前模型ID
        int modelId = this->getModelId();
        auto& model = models_[modelId];

        // 调用 infer 方法进行推理
        model->infer(inputData);

        // 等待数据更新
        resultType result;
        {
            std::unique_lock<std::mutex> resultLock(model->resultMtx_); // 使用 resultMtx_ 锁
            model->cv_.wait(resultLock, [model] { return model->dataReady_; });
            // model->cv_.wait_for(resultLock, std::chrono::milliseconds(1000), [model] { return model->dataReady_; });

            // 重置数据状态
            model->dataReady_ = false;

            // 获取结果
            result = model->getResult(); // 在锁的作用域内获取结果
        }

        // 将帧ID存储到结果中
        resultQueue_.setResult(frameID, result, ID);
    }));

    return 0;
}

// template <typename rknnModel, typename inputType, typename resultType>
// int rknnPool<rknnModel, inputType, resultType>::get(DetectionResult& result) {
//     // std::lock_guard<std::mutex> lock(outputQueueMtx);
//     // if (outputs.empty()) {
//     //     return -1;  // 没有可用结果
//     // }
//     // outputData = std::move(outputs.front());
//     // outputs.pop();
//     return 0;
// }

template <typename rknnModel, typename inputType, typename resultType>
rknnPool<rknnModel, inputType, resultType>::~rknnPool() {
    // pool.reset();
}
