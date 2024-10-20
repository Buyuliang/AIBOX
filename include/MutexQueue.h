#ifndef MUTEXQUEUE_H
#define MUTEXQUEUE_H

#include <unordered_map>
#include <mutex>
#include <vector>
#include <opencv2/core.hpp>
#include <DetectionModels.h>

struct ImageData {
    uint64_t frameID; // 帧ID
    cv::Mat frame; // 原始图像
};

struct FrameData {
    ImageData imageData; // 包含帧ID和图像
    PerDetResult perDetResult;
    PerAttrResult perAttrResult;
    FallDetResult fallDetResult;
    FireSmokeDetResult fireSmokeDetResult;
};

class ImageDataQueue {
public:
    explicit ImageDataQueue(size_t capacity) : capacity_(capacity), head_(0), size_(0) {
        queue_.resize(capacity);
    }

    void push(uint64_t frameID, const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_[head_].frameID = frameID; // 设置帧ID
        queue_[head_].frame = frame.clone(); // 深拷贝图像
        idMap_[frameID] = head_; // 更新映射
        head_ = (head_ + 1) % capacity_; // 更新头指针

        if (size_ < capacity_) {
            size_++; // 队列未满，增加大小
        } else {
            idMap_.erase(queue_[(head_ + capacity_ - 1) % capacity_].frameID); // 移除旧数据的 ID 映射
        }
    }

    void pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) return; // 如果队列为空，直接返回
        idMap_.erase(queue_[tail_].frameID); // 移除旧数据的 ID 映射
        tail_ = (tail_ + 1) % capacity_; // 更新尾指针
        size_--;
    }

    ImageData* front() {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ > 0 ? &queue_[tail_] : nullptr; // 返回第一个元素的指针
    }

    ImageData* getFrameData(uint64_t frameID) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = idMap_.find(frameID);
        return it != idMap_.end() ? &queue_[it->second] : nullptr; // 返回对应元素的指针
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0; // 返回队列是否为空
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear(); // 清空队列
        idMap_.clear(); // 清空映射
        size_ = 0; // 重置大小
        head_ = 0; // 重置头指针
        tail_ = 0; // 重置尾指针
        queue_.resize(capacity_); // 重新调整队列大小
    }

private:
    std::vector<ImageData> queue_; // 存储数据的循环队列
    std::unordered_map<uint64_t, size_t> idMap_; // ID 到索引的映射
    size_t capacity_; // 队列最大容量
    size_t head_; // 当前插入位置
    size_t tail_ = 0; // 当前移除位置
    size_t size_; // 当前队列大小
    mutable std::mutex mutex_; // 保护队列和映射的互斥锁
};

class MutexQueue {
public:
    MutexQueue(size_t capacity) : capacity_(capacity), head_(0), size_(0) {
        queue_.resize(capacity);
    }

    void push(uint64_t frameID, const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        FrameData frameData;
        frameData.imageData.frameID = frameID; // 设置帧ID
        frameData.imageData.frame = frame.clone(); // 深拷贝图像
        queue_[head_] = frameData; // 替换当前位置的数据
        idMap_[frameID] = head_; // 更新映射
        head_ = (head_ + 1) % capacity_; // 更新头指针

        if (size_ < capacity_) {
            size_++; // 队列未满，增加大小
        } else {
            idMap_.erase(queue_[(head_ + capacity_ - 1) % capacity_].imageData.frameID); // 移除旧数据的 ID 映射
        }
    }

    void pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) return; // 如果队列为空，直接返回
        idMap_.erase(queue_[tail_].imageData.frameID); // 移除旧数据的 ID 映射
        tail_ = (tail_ + 1) % capacity_; // 更新尾指针
        size_--;
    }

    FrameData* front() {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ > 0 ? &queue_[tail_] : nullptr; // 返回第一个元素的指针
    }

    FrameData* getFrameData(uint64_t frameID) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = idMap_.find(frameID);
        return it != idMap_.end() ? &queue_[it->second] : nullptr; // 返回对应 FrameData 的指针
    }

    FrameData* setResult(uint64_t frameID, const std::variant<PerDetResult, PerAttrResult, FallDetResult, FireSmokeDetResult>& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = idMap_.find(frameID);
        if (it != idMap_.end()) {
            FrameData& frameData = queue_[it->second];
            std::visit([&frameData](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, PerDetResult>) {
                    frameData.perDetResult = arg; // 修改人检测结果
                } else if constexpr (std::is_same_v<T, PerAttrResult>) {
                    frameData.perAttrResult = arg; // 修改人属性检测结果
                } else if constexpr (std::is_same_v<T, FallDetResult>) {
                    frameData.fallDetResult = arg; // 修改跌倒检测结果
                } else if constexpr (std::is_same_v<T, FireSmokeDetResult>) {
                    frameData.fireSmokeDetResult = arg; // 修改火焰烟雾检测结果
                }
            }, result);
            return &frameData; // 返回修改后的数据指针
        }
        return nullptr;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0; // 返回队列是否为空
    }

    // 清空队列
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear(); // 清空队列
        idMap_.clear(); // 清空映射
        size_ = 0; // 重置大小
        head_ = 0; // 重置头指针
        tail_ = 0; // 重置尾指针
        queue_.resize(capacity_); // 重新调整队列大小
    }

private:
    std::vector<FrameData> queue_; // 存储数据的循环队列
    std::unordered_map<uint64_t, size_t> idMap_; // ID 到索引的映射
    size_t capacity_; // 队列最大容量
    size_t head_; // 当前插入位置
    size_t tail_ = 0; // 当前移除位置
    size_t size_; // 当前队列大小
    mutable std::mutex mutex_; // 保护队列和映射的互斥锁
};

#endif // MUTEXQUEUE_H
