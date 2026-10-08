#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

namespace helipad {

template <class T>
class LatestSlot {
public:
    bool submit(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) return false;
        if (pending_) ++replaced_;
        pending_ = std::move(value);
        cv_.notify_one();
        return true;
    }

    bool tryTake(T& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pending_) return false;
        out = std::move(*pending_);
        pending_.reset();
        return true;
    }

    bool waitTake(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stopped_ || pending_.has_value(); });
        if (!pending_) return false;
        out = std::move(*pending_);
        pending_.reset();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        pending_.reset();
        cv_.notify_all();
    }

    bool hasPending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.has_value();
    }

    std::size_t replacedCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return replaced_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<T> pending_;
    std::size_t replaced_ = 0;
    bool stopped_ = false;
};

} // namespace helipad
