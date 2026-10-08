#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include <opencv2/opencv.hpp>

#include "pureh_core5/detect_logic_isolated_h_revision10_fluent.hpp"

namespace helipad {

// The external lane is metadata only. CoreLock05 remains the sole identity / lock
// authority. Classical Aryan geometry and AI are allowed to nominate ROIs, but
// every nominated ROI is independently re-verified by the PureH verifier and
// still has to pass CoreLock's normal multi-observation confirmation.
enum class CoreLock05HintLane {
    None,
    PureH,
    Plain,
    Far,
    Oblique,
    Generic,
    Ai
};

struct CoreLock05PrimaryResult {
    enum class State { Search, Track, Recover };

    State state = State::Search;
    bool owned = false;
    bool guidanceValid = false;
    bool observedThisFrame = false;
    bool predictedThisFrame = false;

    cv::Rect2f box{};
    cv::Point2f center{};

    float confidence = 0.0f;
    float shapeScore = 0.0f;
    float topologyScore = 0.0f;
    float span = 0.0f;
    bool farMode = false;
    bool provisionalFar = false;

    int confirm = 0;
    int lost = 0;
    int identityMiss = 0;
    int candidateCount = 0;

    CoreLock05HintLane sourceLane = CoreLock05HintLane::None;
    std::string quality = "LOST";
    std::string searchMode = "GLOBAL";
    std::string className = "H";
};

class CoreLock05Primary {
public:
    CoreLock05Primary() { reset(); }

    void reset() {
        tracker_ = std::make_unique<::SearchTrack>(TRACK_REVERIFY_EVERY_DEFAULT);
        motionEstimator_ = std::make_unique<isolated_h_detail::GlobalMotionEstimator>();
        hintVerifier_ = std::make_unique<isolated_h_detail::IsolatedHDetector>();
        motionFrameCounter_ = 0;
        processCounter_ = 0;
        pendingHintLane_ = CoreLock05HintLane::None;
        activeHintLane_ = CoreLock05HintLane::None;
        pendingHintAge_ = 1000;
    }

    bool searching() const {
        return !tracker_ || tracker_->state() == ::SearchTrack::State::SEARCH;
    }

    bool wantsExternalProposal() const {
        return tracker_ &&
            tracker_->state() == ::SearchTrack::State::SEARCH &&
            !tracker_->hasPendingCandidate();
    }

    // Classical proposal bridge. The Aryan lane does NOT create identity. Its
    // candidate is re-verified by PureH's target-conditioned verifier. Because
    // this is two independent geometry proofs (Aryan + PureH), the legacy
    // boresight prior is not allowed to reject an otherwise valid off-centre H;
    // CoreLock's multi-frame topology / clutter / FAR confirmation still applies.
    bool offerGeometryHint(
        const cv::Mat& frame,
        const cv::Point2f& center,
        const cv::Rect2f& rawBox,
        CoreLock05HintLane lane)
    {
        if (!tracker_ || !hintVerifier_ || frame.empty() || !wantsExternalProposal())
            return false;
        if (lane == CoreLock05HintLane::None || lane == CoreLock05HintLane::Ai)
            return false;

        cv::Mat gray;
        if (!toGray(frame, gray)) return false;

        const cv::Rect2f bounded = clampRectF(rawBox, frame.size());
        const cv::Size2f expected(
            std::max(6.0f, bounded.width),
            std::max(6.0f, bounded.height));

        ::DetectionOutput verified = hintVerifier_->trackKnownTarget(
            gray, center, expected, false);

        const ::Candidate* best = chooseVerifiedNear(
            verified, center, std::max(expected.width, expected.height));

        // Some FAR / strongly oblique Aryan boxes are deliberately tight. If
        // target-conditioned verification cannot resolve them, retry one bounded
        // cold PureH pass around the proposal. This still cannot create identity
        // without an actual PureH hVerified candidate.
        ::DetectionOutput fallback;
        if (!best) {
            const cv::Rect2f expanded = expandAndClamp(
                bounded.area() > 0.0f ? bounded : cv::Rect2f(
                    center.x - 12.0f, center.y - 12.0f, 24.0f, 24.0f),
                2.0f,
                frame.size());
            const cv::Rect roi = toIntRect(expanded, frame.size());
            if (roi.width >= 8 && roi.height >= 8) {
                fallback = hintVerifier_->detect(gray, roi, false, false, 0.0, false);
                best = chooseVerifiedNear(
                    fallback, center, std::max(expected.width, expected.height));
            }
        }

        if (!best) return false;

        const bool accepted = tracker_->offerExternalVerifiedCandidate(
            gray, *best, true /* dual-geometry proof: no boresight veto */);
        if (accepted) {
            pendingHintLane_ = lane;
            pendingHintAge_ = 0;
        }
        return accepted;
    }

    // AI assistance remains ROI nomination only. AI is intentionally NOT
    // boresight-exempt: it still needs PureH verification and the normal CoreLock
    // cold-acquisition prior / multi-observation confirmation.
    bool offerAiHint(const cv::Mat& frame, const cv::Rect2f& hintBox) {
        if (!tracker_ || !hintVerifier_ || frame.empty() || !wantsExternalProposal())
            return false;

        cv::Mat gray;
        if (!toGray(frame, gray)) return false;

        const cv::Rect2f expanded = expandAndClamp(hintBox, 1.35f, frame.size());
        const cv::Rect roi = toIntRect(expanded, frame.size());
        if (roi.width < 8 || roi.height < 8) return false;

        ::DetectionOutput verified = hintVerifier_->detect(
            gray, roi, false, false, 0.0, false);

        const ::Candidate* best = nullptr;
        for (const auto& c : verified.candidates) {
            if (!c.hVerified) continue;
            if (!best || c.finalScore > best->finalScore) best = &c;
        }
        if (!best) return false;

        const bool accepted = tracker_->offerExternalVerifiedCandidate(
            gray, *best, false /* keep CoreLock cold-acquisition prior */);
        if (accepted) {
            pendingHintLane_ = CoreLock05HintLane::Ai;
            pendingHintAge_ = 0;
        }
        return accepted;
    }

    CoreLock05PrimaryResult process(const cv::Mat& frame) {
        CoreLock05PrimaryResult result;
        if (!tracker_ || !motionEstimator_ || frame.empty()) return result;

        cv::Mat gray;
        if (!toGray(frame, gray)) return result;

        ++processCounter_;
        if (pendingHintAge_ < 1000) ++pendingHintAge_;

        const ::SearchTrack::State before = tracker_->state();

        ++motionFrameCounter_;
        const bool highRateMotion = tracker_->needsHighRateMotion();
        const bool refreshMotion = highRateMotion ||
            (motionFrameCounter_ % isolated_h_detail::MOTION_EVERY_N_FRAMES) == 1;
        const isolated_h_detail::FrameMotion motion =
            motionEstimator_->update(gray, refreshMotion);

        const ::SearchTrack::Output out = tracker_->update(gray, motion, true);

        if (before == ::SearchTrack::State::SEARCH &&
            out.state == ::SearchTrack::State::TRACK)
        {
            // An external hint only owns the display/source metadata if the lock
            // followed promptly. Otherwise the internal PureH search gets credit.
            if (pendingHintLane_ != CoreLock05HintLane::None && pendingHintAge_ <= 8)
                activeHintLane_ = pendingHintLane_;
            else
                activeHintLane_ = CoreLock05HintLane::PureH;
        }
        else if (out.state == ::SearchTrack::State::SEARCH &&
                 before != ::SearchTrack::State::SEARCH)
        {
            activeHintLane_ = CoreLock05HintLane::None;
            pendingHintLane_ = CoreLock05HintLane::None;
            pendingHintAge_ = 1000;
        }
        else if (out.state == ::SearchTrack::State::SEARCH && pendingHintAge_ > 8) {
            pendingHintLane_ = CoreLock05HintLane::None;
        }

        switch (out.state) {
            case ::SearchTrack::State::SEARCH:
                result.state = CoreLock05PrimaryResult::State::Search;
                break;
            case ::SearchTrack::State::TRACK:
                result.state = CoreLock05PrimaryResult::State::Track;
                break;
            case ::SearchTrack::State::RECOVER:
                result.state = CoreLock05PrimaryResult::State::Recover;
                break;
        }

        result.owned =
            out.state != ::SearchTrack::State::SEARCH && out.box.area() > 0.0f;
        result.guidanceValid = out.guidanceValid;
        result.observedThisFrame = out.observedThisFrame;
        result.predictedThisFrame = out.predictedThisFrame;
        result.box = out.box;
        result.center = out.center;
        result.confidence = static_cast<float>(out.activeConfidence);
        result.shapeScore = static_cast<float>(out.lastCandidate.hScore);
        result.topologyScore = static_cast<float>(
            out.lastCandidate.acquisitionTopologyScore);
        result.span = std::max(out.box.width, out.box.height);
        result.farMode = out.lastCandidate.farMode;
        result.provisionalFar =
            out.lastCandidate.farMode &&
            out.lastCandidate.farSignatureKind ==
                corelock05_policy::FarSignatureKind::Ambiguous;
        result.confirm = out.confirm;
        result.lost = out.lost;
        result.identityMiss = out.identityMiss;
        result.candidateCount = out.candidateCount;
        result.quality = out.quality;
        result.searchMode = out.searchMode;
        result.className = out.lastCandidate.className;
        result.sourceLane = result.owned
            ? (activeHintLane_ == CoreLock05HintLane::None
                ? CoreLock05HintLane::PureH
                : activeHintLane_)
            : CoreLock05HintLane::None;
        return result;
    }

private:
    static bool toGray(const cv::Mat& frame, cv::Mat& gray) {
        if (frame.channels() == 1) {
            gray = frame;
            return true;
        }
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            return true;
        }
        if (frame.channels() == 4) {
            cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
            return true;
        }
        return false;
    }

    static cv::Rect2f clampRectF(const cv::Rect2f& box, const cv::Size& size) {
        const float x1 = std::clamp(box.x, 0.0f, static_cast<float>(size.width));
        const float y1 = std::clamp(box.y, 0.0f, static_cast<float>(size.height));
        const float x2 = std::clamp(box.x + box.width, 0.0f, static_cast<float>(size.width));
        const float y2 = std::clamp(box.y + box.height, 0.0f, static_cast<float>(size.height));
        if (x2 <= x1 || y2 <= y1) return {};
        return {x1, y1, x2 - x1, y2 - y1};
    }

    static cv::Rect toIntRect(const cv::Rect2f& box, const cv::Size& size) {
        cv::Rect roi(
            static_cast<int>(std::floor(box.x)),
            static_cast<int>(std::floor(box.y)),
            std::max(1, static_cast<int>(std::ceil(box.width))),
            std::max(1, static_cast<int>(std::ceil(box.height))));
        roi &= cv::Rect(0, 0, size.width, size.height);
        return roi;
    }

    static cv::Rect2f expandAndClamp(
        const cv::Rect2f& box,
        float factor,
        const cv::Size& size)
    {
        if (box.area() <= 0.0f || size.width <= 0 || size.height <= 0)
            return {};

        const cv::Point2f c(
            box.x + 0.5f * box.width,
            box.y + 0.5f * box.height);
        const float w = std::max(8.0f, box.width * factor);
        const float h = std::max(8.0f, box.height * factor);
        const float x1 = std::max(0.0f, c.x - 0.5f * w);
        const float y1 = std::max(0.0f, c.y - 0.5f * h);
        const float x2 = std::min(static_cast<float>(size.width), c.x + 0.5f * w);
        const float y2 = std::min(static_cast<float>(size.height), c.y + 0.5f * h);
        if (x2 <= x1 || y2 <= y1) return {};
        return {x1, y1, x2 - x1, y2 - y1};
    }

    static const ::Candidate* chooseVerifiedNear(
        const ::DetectionOutput& detection,
        const cv::Point2f& expectedCenter,
        float expectedSpan)
    {
        const ::Candidate* best = nullptr;
        double bestRank = -std::numeric_limits<double>::infinity();
        const double radius = std::max(24.0, 2.0 * static_cast<double>(expectedSpan));
        for (const auto& c : detection.candidates) {
            if (!c.hVerified) continue;
            const double d = cv::norm(c.center - expectedCenter);
            if (d > radius) continue;
            const double spatial = std::max(0.0, 1.0 - d / radius);
            const double rank = c.finalScore + 0.16 * spatial;
            if (rank > bestRank) {
                bestRank = rank;
                best = &c;
            }
        }
        return best;
    }

    std::unique_ptr<::SearchTrack> tracker_;
    std::unique_ptr<isolated_h_detail::GlobalMotionEstimator> motionEstimator_;
    std::unique_ptr<isolated_h_detail::IsolatedHDetector> hintVerifier_;
    std::uint64_t motionFrameCounter_ = 0;
    std::uint64_t processCounter_ = 0;
    CoreLock05HintLane pendingHintLane_ = CoreLock05HintLane::None;
    CoreLock05HintLane activeHintLane_ = CoreLock05HintLane::None;
    int pendingHintAge_ = 1000;
};

} // namespace helipad
