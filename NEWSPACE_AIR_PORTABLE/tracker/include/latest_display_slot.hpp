#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>

namespace helipad {

template <typename T>
class LatestDisplaySlot {
public:
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = false;
        hasItem_ = false;
        replaced_ = 0;
    }

    bool push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) return false;
            if (hasItem_) ++replaced_;
            item_ = std::move(item);
            hasItem_ = true;
        }
        cv_.notify_one();
        return true;
    }

    bool waitPop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return stopped_ || hasItem_; });
        if (!hasItem_) return false;
        out = std::move(item_);
        hasItem_ = false;
        return true;
    }

    bool tryPop(T& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hasItem_) return false;
        out = std::move(item_);
        hasItem_ = false;
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    std::size_t replaced() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return replaced_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    T item_{};
    bool hasItem_ = false;
    bool stopped_ = false;
    std::size_t replaced_ = 0;
};

} // namespace helipad
