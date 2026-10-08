#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

namespace helipad {

enum class CandidateSource {
    None,
    PlainH,
    FarH,
    RingH,
    AiVerifiedH,
    GenericH,
    ObliqueH
};

inline const char* sourceName(CandidateSource s) {
    switch (s) {
        case CandidateSource::PlainH: return "PLAIN_H";
        case CandidateSource::FarH: return "FAR_H";
        case CandidateSource::RingH: return "RING_H";
        case CandidateSource::AiVerifiedH: return "AI_ASSIST_H";
        case CandidateSource::GenericH: return "COLOR_H";
        case CandidateSource::ObliqueH: return "OBLIQUE_H";
        default: return "NONE";
    }
}

enum class HMode { Acquire, Confirm, Track, Recovery, Far };

struct VisionConfig {
    // V10 tracking constants.
    int max_lost_frames = 15;
    float velocity_alpha = 0.65f;
    float center_alpha = 0.65f;
    float box_alpha = 0.55f;

    // LK optical flow - translated from the successful Python V10.
    int lk_win_size = 51;
    int lk_max_level = 4;
    int lk_iterations = 12;
    double lk_epsilon = 0.03;
    int min_flow_points = 3;

    // Plain H acquisition / confirmation / tracking.
    int h_global_search_interval = 4;
    int h_global_watch_interval = 8;
    int h_global_max_side = 1024;
    int h_local_max_side = 900;
    int h_takeover_hits = 2;
    int h_global_confirm_hits = 2;
    int h_global_confirm_interval = 2;
    float h_takeover_score = 0.82f;
    float h_takeover_margin = 0.055f;
    int h_dino_fallback_delay = 90;
    int h_dino_fallback_interval = 90;
    int h_identity_miss_limit = 8;
    int h_recovery_after_misses = 5;
    int h_verify_interval = 2;

    // V15.38 wall-clock identity authority.  These values are deliberately
    // independent of source FPS and processed FPS.
    int locked_identity_period_ms = 110;
    int locked_identity_urgent_period_ms = 45;
    int identity_soft_lease_ms = 320;
    int identity_hard_lease_ms = 720;
    int identity_result_max_age_ms = 460;
    int locked_ring_period_ms = 700;
    int ring_label_lease_ms = 1800;

    // V15.39 non-weakening fast ObliqueH confirmation.
    // These rules are an ADDITIONAL accelerator only.  The original V15.38
    // confirmation path remains intact when any authority test is not met.
    int oblique_fast_required_hits = 2;
    int oblique_fast_min_gap_ms = 55;
    int oblique_fast_max_gap_ms = 320;
    int oblique_fast_verify_period_ms = 90;
    float oblique_fast_min_score = 0.85f;
    float oblique_fast_min_topology = 0.93f;
    float oblique_fast_min_rail_avg = 0.50f;
    float oblique_fast_min_rail = 0.34f;
    float oblique_fast_min_crossbar = 0.42f;
    float oblique_fast_min_void = 0.55f;
    float oblique_fast_min_contrast = 82.0f;
    float oblique_fast_authority_min = 0.72f;
    float oblique_fast_continuity_factor = 0.70f;
    float oblique_fast_scale_ratio_min = 0.55f;
    float oblique_fast_scale_ratio_max = 1.85f;

    // While a non-authoritative ObliqueH is being confirmed, periodically run
    // an independent full-frame direct search so a stronger real H cannot be
    // hidden by a weak cloud/runway proposal.
    int confirm_competition_delay_ms = 180;
    int confirm_competition_period_ms = 260;
    float confirm_competition_margin = 0.08f;

    // H-guided ring upgrade is useful classification evidence, but it must not
    // stall every second confirming frame.  RingH acquisition itself remains
    // fully independent in SEARCHING.
    int confirming_ring_assist_period_ms = 420;

    // V15.40 RingH temporal authority.  This is an ADDITIVE confirmation path
    // for a RingH candidate already returned by the existing fast ring geometry
    // lane.  No H/Ring detector threshold is changed.  If this path is not
    // satisfied, the V15.39/V15.38 confirmation logic remains available.
    int ring_fast_required_hits = 2;
    int ring_fast_required_h_hits = 2;
    int ring_fast_min_gap_ms = 70;
    // V15.41: confirmation must remain possible even when a heavy fallback
    // temporarily drives tracker.process() below 3 FPS.  Same-target and scale
    // guards still apply, so this widens only the temporal observation window.
    int ring_fast_max_gap_ms = 1800;
    int ring_fast_verify_period_ms = 140;
    // DEEP_RING stays available, but only after fast ring evidence has gone
    // stale.  Re-running DEEP_RING while RFAST is alive was the main cause of
    // the 1-5 FPS confirmation collapse seen in 11.webm.
    int ring_deep_confirm_period_ms = 900;
    int ring_deep_stale_after_fast_ms = 1400;
    // The strict H verifier needs surrounding structure; the H-only target box
    // can be too tight inside a ring.  Search a central fraction of the ring
    // first, which removes most of the circle edge while preserving the H.
    float ring_h_core_scale = 0.72f;
    int ring_h_fallback_min_ms = 160;
    float ring_fast_min_pair = 0.28f;
    float ring_fast_continuity_factor = 0.90f;
    float ring_fast_scale_ratio_min = 0.50f;
    float ring_fast_scale_ratio_max = 2.00f;

    int h_confirm_hits = 3;
    int h_confirm_miss_limit = 3;
    int h_confirm_timeout = 18;

    // Dedicated V10 long-range path.
    bool h_far_enabled = true;
    int h_far_search_interval = 4;
    int h_far_confirm_interval = 4;
    int h_far_grace_frames = 30;
    int h_far_confirm_timeout = 32;
    int h_far_confirm_miss_limit = 4;
    std::vector<int> h_far_scale_sides{620, 760, 900, 1040, 1180, 1320};
    int h_far_candidates_per_scale = 64;
    int h_far_native_verify_limit = 16;
    int h_far_required_hits = 3;
    float h_far_min_raw_span = 5.5f;
    float h_far_max_raw_span = 90.0f;
    float h_far_acquire_score = 0.88f;
    float h_far_confirm_score = 0.80f;
    float h_far_match_radius = 62.0f;
    float h_far_x1 = 0.22f;
    float h_far_x2 = 0.78f;
    float h_far_y1 = 0.08f;
    float h_far_y2 = 0.80f;

    // V14 perspective / oblique H path.  This is deliberately separate from
    // the proven tiny FAR_H path: it is for a marker that is already tens of
    // pixels across but whose projective skew makes one rail look shorter.
    bool h_oblique_enabled = true;
    int h_oblique_search_interval = 4;
    int h_oblique_confirm_interval = 2;
    int h_oblique_required_hits = 2;
    int h_oblique_confirm_miss_limit = 4;
    int h_oblique_confirm_timeout = 24;
    std::vector<int> h_oblique_scale_sides{1000, 1150};
    float h_oblique_x1 = 0.18f;
    float h_oblique_x2 = 0.82f;
    float h_oblique_y1 = 0.08f;
    float h_oblique_y2 = 0.92f;
    float h_oblique_min_raw_span = 25.0f;
    float h_oblique_max_raw_span = 340.0f;
    float h_oblique_acquire_score = 0.76f;
    float h_oblique_confirm_score = 0.70f;

    // Healthy-lock guard: a different full-frame H cannot steal a good lock.
    int takeover_identity_misses = 3;
    int takeover_lost_frames = 2;

    // V10 ring helper. It is never an aim point by itself.
    bool universal_ring_enabled = true;
    int universal_ring_interval = 4;
    int universal_ring_max_side = 1100;
    float universal_ring_min_axis = 10.0f;
    float universal_ring_max_aspect = 3.25f;
    float universal_ring_fit_residual = 0.20f;
    float universal_ring_pair_center = 0.20f;
    float universal_ring_h_min = 0.70f;
    float universal_ring_single_h_min = 0.90f;
    float universal_local_h_min = 0.70f;
    int ring_confirm_interval = 4;
    int ring_confirm_miss_limit = 5;
    int ring_confirm_timeout = 60;

    // Color/polarity invariant H is local/AI/ring fallback only.
    bool universal_color_enabled = true;
    bool universal_color_global_search = false;
    int universal_color_max_side = 1150;
    float universal_color_h_min = 0.80f;

    // AI remains a delayed/recovery helper, never target authority.
    int universal_ai_assist_delay = 30;
    bool ai_escalate_on_confirm_failure = true;
    int ai_escalation_after_failures = 1;
    int ai_escalation_cooldown = 20;
    int ai_top_k = 8;

    // Guidance - exact V10 semantics.
    float roll_gain = 25.0f;
    float pitch_gain = 25.0f;
    float max_roll = 30.0f;
    float max_pitch = 30.0f;
};

struct Candidate {
    cv::Point2f center{};
    cv::Rect2f box{};
    cv::Rect2f rawBox{};
    float score = 0.0f;
    float shapeScore = 0.0f;
    float topology = 0.0f;
    float span = 0.0f;
    float area = 0.0f;
    float distance = 0.0f;
    float fill = 0.0f;
    float solidity = 0.0f;
    float railMin = 0.0f;
    float railAvg = 0.0f;
    float crossbar = 0.0f;
    float voidTop = 0.0f;
    float voidBottom = 0.0f;
    float centreFill = 0.0f;
    CandidateSource source = CandidateSource::None;
    bool hasRing = false;
    cv::RotatedRect ring{};
    cv::Rect2f ringBox{};
    cv::Point2f ringCenter{};
    float ringPairScore = 0.0f;
    float ringFitScore = 0.0f;
};

struct Guidance {
    float errorX = 0.0f;       // + right, pixels
    float errorY = 0.0f;       // + above, pixels
    float normalizedX = 0.0f;
    float normalizedY = 0.0f;
    float roll = 0.0f;
    float pitch = 0.0f;
};

struct HTopologyMetrics {
    float score = 0.0f;
    bool valid = false;
    float railMin = 0.0f;
    float railAvg = 0.0f;
    float crossbar = 0.0f;
    float voidTop = 0.0f;
    float voidBottom = 0.0f;
    float centreFill = 0.0f;
};

class HGeometryDetector {
public:
    explicit HGeometryDetector(VisionConfig cfg);

    std::optional<Candidate> findH(
        const cv::Mat& bgr,
        std::optional<cv::Rect2f> anchorBox = std::nullopt,
        std::optional<cv::Point2f> predictedCenter = std::nullopt,
        std::optional<cv::Rect2f> predictedBox = std::nullopt,
        HMode mode = HMode::Acquire) const;

    std::optional<Candidate> findFar(
        const cv::Mat& bgr,
        std::optional<cv::Point2f> predictedCenter = std::nullopt,
        std::optional<cv::Rect2f> predictedBox = std::nullopt,
        std::optional<int> scaleSide = std::nullopt) const;

    // Two-scale, affine-shear-tolerant H detector for steep viewing angles.
    // The result is still H geometry; no AI box is allowed to define the aim point.
    std::optional<Candidate> findOblique(
        const cv::Mat& bgr,
        std::optional<cv::Point2f> predictedCenter = std::nullopt,
        std::optional<cv::Rect2f> predictedBox = std::nullopt) const;

    std::optional<Candidate> findColorInvariant(
        const cv::Mat& bgr,
        std::optional<cv::Rect2f> anchorBox = std::nullopt,
        std::optional<cv::Point2f> predictedCenter = std::nullopt,
        std::optional<cv::Rect2f> predictedBox = std::nullopt,
        HMode mode = HMode::Recovery) const;

    HTopologyMetrics topology(const cv::Mat& normalizedMask, HMode mode) const;

private:
    VisionConfig cfg_;

    cv::Rect2f searchBox(const cv::Mat& bgr,
                         std::optional<cv::Rect2f> anchorBox,
                         std::optional<cv::Rect2f> predictedBox,
                         HMode mode) const;
    cv::Mat contrastMask(const cv::Mat& bgrCrop,
                         std::optional<float> expectedSpan,
                         bool globalMode) const;
    cv::Mat farContrastMask(const cv::Mat& bgrCrop) const;
    cv::Mat normalizedComponentMask(const std::vector<cv::Point>& contour) const;
    HTopologyMetrics topologyOneOrientation(const cv::Mat& mask, HMode mode) const;
    std::vector<Candidate> farCandidatesOneScale(
        const cv::Mat& bgr,
        const cv::Rect2f& roi,
        int maxSide,
        std::optional<cv::Point2f> predictedCenter,
        std::optional<float> predictedSpan) const;
    std::optional<Candidate> farNativeVerify(const cv::Mat& bgr,
                                              const Candidate& candidate) const;
};

class RingHelper {
public:
    RingHelper(VisionConfig cfg, const HGeometryDetector& hDetector);

    std::optional<Candidate> findRingH(
        const cv::Mat& bgr,
        std::optional<cv::Rect2f> anchorBox = std::nullopt,
        std::optional<cv::Point2f> predictedCenter = std::nullopt,
        std::optional<cv::Rect2f> predictedBox = std::nullopt,
        HMode mode = HMode::Acquire) const;

private:
    VisionConfig cfg_;
    const HGeometryDetector& hDetector_;
};

class OpticalTracker {
public:
    bool seed(const cv::Mat& gray, const cv::Rect2f& box);
    bool update(const cv::Mat& previousGray, const cv::Mat& gray,
                const cv::Rect2f& targetBox, const cv::Point2f& velocity,
                cv::Point2f& delta, float& quality);
    void clear();
    bool ready() const { return static_cast<int>(points_.size()) >= 3; }

private:
    std::vector<cv::Point2f> points_;
};

Guidance calculateGuidance(const cv::Point2f& target, int width, int height,
                           const VisionConfig& cfg);
cv::Rect2f clampRect(const cv::Rect2f& r, int width, int height);
cv::Rect2f expandRect(const cv::Rect2f& r, float factor, int width, int height);
cv::Rect2f moveRect(const cv::Rect2f& r, float dx, float dy);
float rectSpan(const cv::Rect2f& r);
cv::Point2f rectCenter(const cv::Rect2f& r);

} // namespace helipad
