#pragma once

#include "vision_shm.hpp"
#include <opencv2/core.hpp>

#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <time.h>

namespace remote_display_shm {

inline std::uint64_t monotonicNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

inline std::uint64_t sessionId() {
    return (monotonicNs() << 11U) ^ static_cast<std::uint64_t>(::getpid()) ^ 0x4453504c41595348ULL;
}

class Publisher {
public:
    Publisher() = default;
    Publisher(const Publisher&) = delete;
    Publisher& operator=(const Publisher&) = delete;
    ~Publisher() { close(); }

    bool open(const std::string& name) {
        close();
        if (!vision_shm::validShmName(name)) {
            std::cerr << "[REMOTE-DISPLAY] invalid SHM name: " << name << "\n";
            return false;
        }
        fd_ = ::shm_open(name.c_str(), O_CREAT | O_RDWR, 0660);
        if (fd_ < 0) {
            std::cerr << "[REMOTE-DISPLAY] shm_open failed: " << std::strerror(errno) << "\n";
            return false;
        }
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            std::cerr << "[REMOTE-DISPLAY] another publisher owns " << name << "\n";
            close();
            return false;
        }
        if (::ftruncate(fd_, static_cast<off_t>(vision_shm::regionBytes())) != 0) {
            std::cerr << "[REMOTE-DISPLAY] ftruncate failed: " << std::strerror(errno) << "\n";
            close();
            return false;
        }
        void* p = ::mmap(nullptr, vision_shm::regionBytes(), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) {
            std::cerr << "[REMOTE-DISPLAY] mmap failed: " << std::strerror(errno) << "\n";
            close();
            return false;
        }
        region_ = static_cast<vision_shm::SharedRegion*>(p);
        vision_shm::initializeRegion(region_, sessionId());
        name_ = name;
        sequence_ = 0;
        return true;
    }

    bool publish(const cv::Mat& bgr) {
        if (!region_ || bgr.empty() || bgr.type() != CV_8UC3) return false;
        if (bgr.cols > vision_shm::kMaxWidth || bgr.rows > vision_shm::kMaxHeight) return false;
        cv::Mat packed;
        const cv::Mat* src = &bgr;
        if (!bgr.isContinuous() || bgr.step != static_cast<std::size_t>(bgr.cols * 3)) {
            packed = bgr.clone();
            src = &packed;
        }
        const int stride = static_cast<int>(src->step);
        const std::size_t bytes = src->step * static_cast<std::size_t>(src->rows);
        return vision_shm::commitFrame(region_, src->data, bytes, src->cols, src->rows,
                                       stride, 3, monotonicNs(), ++sequence_);
    }

    bool active() const { return region_ != nullptr; }
    const std::string& name() const { return name_; }

private:
    void close() {
        if (region_) {
            ::munmap(region_, vision_shm::regionBytes());
            region_ = nullptr;
        }
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
            fd_ = -1;
        }
        name_.clear();
    }

    int fd_ = -1;
    vision_shm::SharedRegion* region_ = nullptr;
    std::uint64_t sequence_ = 0;
    std::string name_;
};

} // namespace remote_display_shm
