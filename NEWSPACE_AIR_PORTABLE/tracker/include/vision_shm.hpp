#pragma once

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>

namespace vision_shm {

// PureH /vision_camera ABI. Keep byte-for-byte compatible with the existing
// PureH publisher and consumer so one publisher can feed both programs.
constexpr std::uint64_t kMagic = 0x564953494f4e5348ULL; // "VISIONSH"
constexpr std::uint32_t kVersion = 1u;
constexpr std::uint32_t kSlotCount = 3u;
constexpr int kMaxWidth = 3840;
constexpr int kMaxHeight = 2160;
constexpr int kMaxChannels = 3;
constexpr std::size_t kMaxPayloadBytes =
    static_cast<std::size_t>(kMaxWidth) *
    static_cast<std::size_t>(kMaxHeight) *
    static_cast<std::size_t>(kMaxChannels);
constexpr const char* kDefaultShmName = "/vision_camera";

enum class PixelFormat : std::uint32_t {
    Unknown = 0,
    BGR8 = 1,
};

struct FrameMeta {
    std::uint64_t sequence = 0;
    std::uint64_t captureMonotonicNs = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t stride = 0;
    std::uint32_t channels = 0;
    std::uint32_t payloadBytes = 0;
    std::uint32_t reserved = 0;
};

struct FrameSlot {
    alignas(8) std::uint64_t generation = 0;
    FrameMeta meta{};
    alignas(64) std::uint8_t payload[kMaxPayloadBytes]{};
};

struct SharedRegion {
    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t slotCount = 0;
    std::uint64_t regionBytes = 0;
    std::uint64_t sessionId = 0;
    alignas(8) std::uint64_t publishedSequence = 0;
    alignas(8) std::uint64_t publisherHeartbeatNs = 0;
    std::uint64_t reserved[8]{};
    FrameSlot slots[kSlotCount];
};

inline constexpr std::size_t regionBytes() {
    return sizeof(SharedRegion);
}

inline bool validShmName(const std::string& name) {
    return name.size() >= 2 && name.front() == '/' &&
           name.find('/', 1) == std::string::npos;
}

inline void initializeRegion(SharedRegion* region, std::uint64_t sessionId) {
    if (!region) return;
    region->magic = 0;
    region->version = 0;
    region->slotCount = 0;
    region->regionBytes = sizeof(SharedRegion);
    region->sessionId = sessionId;
    __atomic_store_n(&region->publishedSequence, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&region->publisherHeartbeatNs, 0u, __ATOMIC_RELEASE);
    std::memset(region->reserved, 0, sizeof(region->reserved));
    for (std::uint32_t i = 0; i < kSlotCount; ++i) {
        __atomic_store_n(&region->slots[i].generation, 0u, __ATOMIC_RELEASE);
        region->slots[i].meta = FrameMeta{};
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    region->version = kVersion;
    region->slotCount = kSlotCount;
    region->magic = kMagic;
}

inline bool regionLooksValid(const SharedRegion* region) {
    return region != nullptr &&
           region->magic == kMagic &&
           region->version == kVersion &&
           region->slotCount == kSlotCount &&
           region->regionBytes == sizeof(SharedRegion);
}

inline bool metadataLooksValid(const FrameMeta& meta) {
    if (meta.sequence == 0 || meta.width == 0 || meta.height == 0 ||
        meta.channels == 0 || meta.channels > kMaxChannels || meta.stride == 0) {
        return false;
    }
    if (meta.width > static_cast<std::uint32_t>(kMaxWidth) ||
        meta.height > static_cast<std::uint32_t>(kMaxHeight)) {
        return false;
    }
    const std::uint64_t minimumStride =
        static_cast<std::uint64_t>(meta.width) * meta.channels;
    if (meta.stride < minimumStride) return false;
    const std::uint64_t required =
        static_cast<std::uint64_t>(meta.stride) * meta.height;
    return required == meta.payloadBytes && required <= kMaxPayloadBytes;
}

inline bool commitFrame(
    SharedRegion* region,
    const std::uint8_t* data,
    std::size_t bytes,
    int width,
    int height,
    int stride,
    int channels,
    std::uint64_t captureMonotonicNs,
    std::uint64_t sequence)
{
    if (!regionLooksValid(region) || !data || sequence == 0 ||
        width <= 0 || height <= 0 || stride <= 0 || channels <= 0 ||
        channels > kMaxChannels || width > kMaxWidth || height > kMaxHeight) {
        return false;
    }
    const std::uint64_t required =
        static_cast<std::uint64_t>(stride) * static_cast<std::uint64_t>(height);
    if (required != bytes || required > kMaxPayloadBytes ||
        static_cast<std::uint64_t>(stride) <
            static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(channels)) {
        return false;
    }

    FrameSlot& slot = region->slots[sequence % kSlotCount];
    std::uint64_t generation =
        __atomic_load_n(&slot.generation, __ATOMIC_RELAXED);
    if (generation & 1u) ++generation;
    __atomic_store_n(&slot.generation, generation + 1u, __ATOMIC_RELEASE);

    slot.meta.sequence = sequence;
    slot.meta.captureMonotonicNs = captureMonotonicNs;
    slot.meta.width = static_cast<std::uint32_t>(width);
    slot.meta.height = static_cast<std::uint32_t>(height);
    slot.meta.stride = static_cast<std::uint32_t>(stride);
    slot.meta.channels = static_cast<std::uint32_t>(channels);
    slot.meta.payloadBytes = static_cast<std::uint32_t>(bytes);
    slot.meta.reserved = 0;
    std::memcpy(slot.payload, data, bytes);

    __atomic_store_n(&slot.generation, generation + 2u, __ATOMIC_RELEASE);
    __atomic_store_n(&region->publisherHeartbeatNs, captureMonotonicNs, __ATOMIC_RELEASE);
    __atomic_store_n(&region->publishedSequence, sequence, __ATOMIC_RELEASE);
    return true;
}

inline bool readMetaForSequence(
    const SharedRegion* region,
    std::uint64_t sequence,
    FrameMeta& outMeta)
{
    if (!regionLooksValid(region) || sequence == 0) return false;
    const FrameSlot& slot = region->slots[sequence % kSlotCount];
    const std::uint64_t before =
        __atomic_load_n(&slot.generation, __ATOMIC_ACQUIRE);
    if (before & 1u) return false;
    const FrameMeta meta = slot.meta;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    const std::uint64_t after =
        __atomic_load_n(&slot.generation, __ATOMIC_ACQUIRE);
    if (before != after || (after & 1u) || meta.sequence != sequence ||
        !metadataLooksValid(meta)) {
        return false;
    }
    outMeta = meta;
    return true;
}

inline bool copyFramePacked(
    const SharedRegion* region,
    std::uint64_t sequence,
    std::uint8_t* destination,
    std::size_t destinationCapacity,
    std::size_t destinationStride,
    FrameMeta& outMeta)
{
    if (!regionLooksValid(region) || !destination || sequence == 0) return false;
    const FrameSlot& slot = region->slots[sequence % kSlotCount];
    const std::uint64_t before =
        __atomic_load_n(&slot.generation, __ATOMIC_ACQUIRE);
    if (before & 1u) return false;

    const FrameMeta meta = slot.meta;
    if (meta.sequence != sequence || !metadataLooksValid(meta)) return false;

    const std::size_t rowBytes =
        static_cast<std::size_t>(meta.width) * meta.channels;
    const std::size_t needed =
        destinationStride * static_cast<std::size_t>(meta.height);
    if (destinationStride < rowBytes || needed > destinationCapacity) return false;

    for (std::uint32_t y = 0; y < meta.height; ++y) {
        std::memcpy(
            destination + static_cast<std::size_t>(y) * destinationStride,
            slot.payload + static_cast<std::size_t>(y) * meta.stride,
            rowBytes);
    }

    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    const std::uint64_t after =
        __atomic_load_n(&slot.generation, __ATOMIC_ACQUIRE);
    if (before != after || (after & 1u)) return false;
    outMeta = meta;
    return true;
}

inline std::uint64_t latestSequence(const SharedRegion* region) {
    if (!regionLooksValid(region)) return 0;
    return __atomic_load_n(&region->publishedSequence, __ATOMIC_ACQUIRE);
}

struct FrameInfo {
    std::uint64_t sequence = 0;
    std::uint64_t timestamp_ns = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t stride = 0;
    PixelFormat pixel_format = PixelFormat::Unknown;
};

struct Info {
    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t slot_count = 0;
    PixelFormat pixel_format = PixelFormat::Unknown;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t stride = 0;
    std::uint64_t slot_payload_bytes = 0;
    std::uint64_t mapped_bytes = 0;
    std::uint64_t published_sequence = 0;
    std::uint32_t active_slot = 0;
    std::uint64_t published_timestamp_ns = 0;
};

inline void throwErrno(const std::string& what) {
    throw std::system_error(errno, std::generic_category(), what);
}

class Reader {
public:
    Reader() = default;
    ~Reader() { close(); }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    void open(const std::string& name) {
        close();
        if (!validShmName(name)) {
            throw std::invalid_argument(
                "POSIX shm name must be /name with no second slash");
        }

        fd_ = ::shm_open(name.c_str(), O_RDONLY, 0);
        if (fd_ < 0) throwErrno("shm_open reader(" + name + ")");

        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            const int saved = errno;
            close();
            errno = saved;
            throwErrno("fstat shared memory");
        }
        if (st.st_size < static_cast<off_t>(regionBytes())) {
            const auto actual = st.st_size;
            close();
            std::ostringstream ss;
            ss << "PureH SHM region too small: got " << actual
               << " bytes expected >= " << regionBytes();
            throw std::runtime_error(ss.str());
        }

        void* p = ::mmap(nullptr, regionBytes(), PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) {
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            errno = saved;
            throwErrno("mmap shared memory reader");
        }
        region_ = static_cast<const SharedRegion*>(p);

        if (!regionLooksValid(region_)) {
            const std::uint64_t actual = region_->magic;
            const std::uint32_t version = region_->version;
            const std::uint32_t slots = region_->slotCount;
            close();
            std::ostringstream ss;
            ss << "PureH SHM ABI mismatch: magic=0x"
               << std::hex << actual << std::dec
               << " version=" << version
               << " slots=" << slots
               << "; expected VISIONSH magic=0x"
               << std::hex << kMagic << std::dec
               << " version=" << kVersion
               << " slots=" << kSlotCount;
            throw std::runtime_error(ss.str());
        }
    }

    Info info() const {
        if (!region_) throw std::runtime_error("SHM reader is not open");

        // Aryan's main asks for dimensions immediately after open. PureH stores
        // dimensions per slot, so wait briefly for the first published frame.
        FrameMeta meta{};
        for (int i = 0; i < 400; ++i) {
            const std::uint64_t seq = latestSequence(region_);
            if (seq != 0 && readMetaForSequence(region_, seq, meta)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        Info out;
        out.magic = region_->magic;
        out.version = region_->version;
        out.slot_count = region_->slotCount;
        out.mapped_bytes = region_->regionBytes;
        out.published_sequence = latestSequence(region_);
        out.active_slot = static_cast<std::uint32_t>(
            out.published_sequence % kSlotCount);

        if (metadataLooksValid(meta)) {
            out.pixel_format = meta.channels == 3
                ? PixelFormat::BGR8 : PixelFormat::Unknown;
            out.width = meta.width;
            out.height = meta.height;
            out.stride = meta.stride;
            out.slot_payload_bytes = meta.payloadBytes;
            out.published_timestamp_ns = meta.captureMonotonicNs;
        }
        return out;
    }

    bool readLatestInto(
        std::uint8_t* dst,
        std::size_t dst_bytes,
        std::uint32_t dst_stride,
        FrameInfo& out,
        int retries = 8) const
    {
        if (!region_) throw std::runtime_error("SHM reader is not open");
        if (!dst) throw std::invalid_argument("destination buffer is null");

        for (int attempt = 0; attempt < retries; ++attempt) {
            const std::uint64_t sequence = latestSequence(region_);
            if (sequence == 0) return false;

            FrameMeta meta{};
            if (!readMetaForSequence(region_, sequence, meta)) continue;
            if (meta.channels != 3) {
                throw std::runtime_error("PureH SHM frame is not 3-channel BGR");
            }

            const std::size_t rowBytes =
                static_cast<std::size_t>(meta.width) * meta.channels;
            const std::size_t needed =
                static_cast<std::size_t>(dst_stride) * meta.height;
            if (dst_stride < rowBytes) {
                throw std::invalid_argument("destination stride is too small for BGR8");
            }
            if (needed > dst_bytes) {
                throw std::invalid_argument("destination buffer is too small");
            }

            FrameMeta copied{};
            if (!copyFramePacked(
                    region_, sequence, dst, dst_bytes, dst_stride, copied)) {
                continue;
            }

            out.sequence = copied.sequence;
            out.timestamp_ns = copied.captureMonotonicNs;
            out.width = copied.width;
            out.height = copied.height;
            out.stride = copied.stride;
            out.pixel_format = PixelFormat::BGR8;
            return true;
        }
        return false;
    }

    void close() noexcept {
        if (region_) {
            ::munmap(const_cast<SharedRegion*>(region_), regionBytes());
            region_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool isOpen() const noexcept { return region_ != nullptr; }

private:
    int fd_ = -1;
    const SharedRegion* region_ = nullptr;
};

} // namespace vision_shm
