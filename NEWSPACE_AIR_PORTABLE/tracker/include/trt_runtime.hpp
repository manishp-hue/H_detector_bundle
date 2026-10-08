#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vision.hpp"

namespace helipad {

struct AiProposal {
    cv::Rect2f box;
    float score = 0.0f;
};

class TensorRtAiRecovery {
public:
    TensorRtAiRecovery();
    ~TensorRtAiRecovery();

    // Engines are optional. With no engines, the C++ geometry/tracker remains usable.
    bool load(const std::string& dinoPlan,
              const std::string& samPlan = std::string());
    bool available() const;
    bool samAvailable() const;

    std::vector<AiProposal> detect(const cv::Mat& bgr, int topK);

    // If a SAM engine with the documented contract is available, refine a proposal.
    // Otherwise this returns the original proposal box.
    cv::Rect2f refineWithSam(const cv::Mat& bgr, const cv::Rect2f& proposal,
                             float* maskScore = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace helipad
