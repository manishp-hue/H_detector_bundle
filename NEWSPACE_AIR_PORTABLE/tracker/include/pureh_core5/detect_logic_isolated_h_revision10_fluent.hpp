#pragma once

// detect_logic_isolated_h_revision10_fluent.hpp
//
// Optimized grayscale-only, non-ML H detector for high-speed guidance.
//
// Core policy:
//   * One search task per input frame.
//   * 10-frame SEARCH cadence: 60% SMALL, 30% MEDIUM, 10% GLOBAL.
//   * Far H (about 10-24 px): trusted Rev11 projection + temporal evidence.
//   * Strict H (about 25 px or larger): normalized H geometry + local clearance.
//   * H inside larger geometry: bounded inner-core refinement reuses the same
//     Rev11 H verifier with the measured H-core size for FAR/STRICT selection.
//   * Resolved fresh observations get a negative three-stroke integrity guard.
//   * CoreLock_04: ambiguous FAR blobs need compatible micro-H evidence.
//   * CoreLock_04: active scale/state is canonical H-only, never outer pad support.
//   * CoreLock_04: stale prediction may escape only to a twice-verified resolved H.
//   * CoreLock_05: ambiguous FAR identity needs bridge/leg-compatible topology.
//   * CoreLock_05: ambiguous FAR locks stay provisional until fresh H evidence.
//   * CoreLock_05: nested H-core probing handles H merged with enclosing support.
//   * No circle / ellipse detector: clutter beyond the H clearance halo is ignored.
//   * Once locked, preserve the active target and recover locally.
//   * Switch to another H only after a large active-confidence drop and
//     a persistent stronger challenger.
//
// The loop file owns input, display and guidance coordinates. This header owns
// detection, selection, tracking and recovery.

#include <opencv2/opencv.hpp>
#include "track.hpp"
#include "pure_h_clearance_policy.hpp"
#include "pure_h_corelock_policy.hpp"
#include "pure_h_corelock03_policy.hpp"
#include "pure_h_corelock04_policy.hpp"
#include "pure_h_corelock05_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Kept for CLI compatibility. Identity is evaluated on every scheduled task.
static constexpr int TRACK_REVERIFY_EVERY_DEFAULT = 1;

namespace isolated_h_detail {

// ============================================================
// Configuration
// ============================================================

static constexpr int SCHEDULE_LENGTH = 10;

enum class SearchTask
{
    SMALL,
    MEDIUM,
    GLOBAL
};

// Revision-10 FLUENT: exact 60% / 30% / 10% in a compact 10-frame cycle.
// Spread MEDIUM work instead of placing expensive frames back-to-back:
// S S S M S M S S M G
static constexpr std::array<SearchTask, SCHEDULE_LENGTH> SEARCH_SCHEDULE = {{
    SearchTask::SMALL,
    SearchTask::SMALL,
    SearchTask::SMALL,
    SearchTask::MEDIUM,
    SearchTask::SMALL,
    SearchTask::MEDIUM,
    SearchTask::SMALL,
    SearchTask::SMALL,
    SearchTask::MEDIUM,
    SearchTask::GLOBAL
}};

// Fixed SEARCH ROI dimensions in original-frame pixels.
// These replace percentage-based SMALL / MEDIUM windows.
// Change only these four values to tune the field-of-view searched per task.
static constexpr int SMALL_SEARCH_WIDTH_PX  = 370;
static constexpr int SMALL_SEARCH_HEIGHT_PX = 740;
static constexpr int MEDIUM_SEARCH_WIDTH_PX  = 510;
static constexpr int MEDIUM_SEARCH_HEIGHT_PX = 1020;

// GLOBAL always scans the complete frame (internally proposal work may be
// downscaled for FPS, but the logical search ROI is 100% of the frame).

static constexpr int GLOBAL_MAX_WIDTH = 800;
static constexpr int LOCAL_PROPOSAL_MAX_WIDTH = 800;
static constexpr int TRACK_PROPOSAL_MAX_WIDTH = 640;
static constexpr int SHAPE_WORK_MAX_DIM = 96;
static constexpr int CLEARANCE_WORK_MAX_DIM = 224;
static constexpr int CORE_REFINE_WORK_MAX_DIM = 160;
static constexpr int CORE_REFINE_MAX_CANDIDATES = 2;
static constexpr int CORE_REFINE_MAX_PROPOSALS_PER_PASS = 2;
static constexpr double CORE_REFINE_SCALE_MISMATCH_RATIO = 0.72;
static constexpr double PURE_H_CLEARANCE_FRACTION = 0.05;
static constexpr int MAX_PROPOSALS_LOCAL = 10;
static constexpr int MAX_PROPOSALS_TRACK = 6;
static constexpr int MAX_PROPOSALS_GLOBAL = 14;
static constexpr int MAX_ACCEPTED = 8;

// Minimum physical image support.
static constexpr double FAR_MIN_MAJOR_PX = 10.0;
static constexpr double FAR_MIN_MINOR_PX = 4.0;
static constexpr double STRICT_MIN_MAJOR_PX = 24.0;
static constexpr double STRICT_MIN_MINOR_PX = 8.0;
static constexpr double MAX_TARGET_FRACTION = 0.75;

// Identity gates.
static constexpr double FAR_SHAPE_MIN = 0.72;
static constexpr double STRICT_SHAPE_MIN = 0.74;
static constexpr double STRICT_FAST_SHAPE = 0.88;
static constexpr double STRICT_FAST_QUIET = 0.84;
static constexpr double STRICT_FAST_CONF = 0.85;

// Initial drone-view acquisition is boresight-prioritized.  An off-centre
// candidate must be exceptionally H-like.
static constexpr double INITIAL_CENTER_SCORE_MIN = 0.74;
static constexpr double OFFCENTER_H_OVERRIDE = 0.92;

// Acquisition.
static constexpr int STRICT_CONFIRM_FRAMES = 2;
static constexpr int FAR_CONFIRM_FRAMES = 4;
static constexpr int PENDING_MAX_AGE = 8;
static constexpr int FOCUS_CONFIRM_FRAMES = 5;

// Tracking / recovery.
static constexpr int PREDICT_GUIDANCE_FRAMES = 0;
static constexpr int RESET_AFTER_LOCAL_MISSES = 10;
static constexpr double POSITION_ALPHA = 0.62;
static constexpr double VELOCITY_BETA = 0.28;
static constexpr double SIZE_ALPHA = 0.40; // retained for compatibility
static constexpr double MISS_CONFIDENCE_DECAY = 0.92;
static constexpr double MIN_PREDICT_CONFIDENCE = 0.55;

// Dynamic target-growth model. After lock, width and height are predicted
// independently so an approaching / perspective-changing H can grow without
// being rejected by a fixed previous-size gate.
static constexpr double SIZE_RATE_ALPHA = 0.34;
static constexpr double WIDTH_HEIGHT_GROW_ALPHA = 0.72;
static constexpr double WIDTH_HEIGHT_SHRINK_ALPHA = 0.28;
static constexpr double MAX_LOG_GROWTH_PER_FRAME = 0.32;
static constexpr double MAX_LOG_SHRINK_PER_FRAME = -0.20;
static constexpr double SIZE_LOOKAHEAD_FRAMES = 2.0;

// Dynamic local-buffer policy. The ROI grows independently in X/Y according
// to the current H width/height, recent growth and image motion. Capping the
// ROI prevents a close marker from consuming the whole frame and hurting FPS.
static constexpr double LOCAL_MARGIN_SCALE = 2.20;
static constexpr double MEDIUM_MARGIN_SCALE = 3.40;
static constexpr double RECOVERY_MARGIN_SCALE = 4.20;
static constexpr double GROWTH_MARGIN_GAIN = 4.5;
static constexpr double MAX_GROWTH_MARGIN_BONUS = 1.60;
static constexpr double MOTION_MARGIN_FRAMES = 2.5;
// Minimum target-centred tracking windows use the same explicit pixel sizes
// as SEARCH. Dynamic growth / motion can expand beyond these minima.
static constexpr int LOCAL_MIN_WIDTH_PX   = SMALL_SEARCH_WIDTH_PX;
static constexpr int LOCAL_MIN_HEIGHT_PX  = SMALL_SEARCH_HEIGHT_PX;
static constexpr int MEDIUM_MIN_WIDTH_PX  = MEDIUM_SEARCH_WIDTH_PX;
static constexpr int MEDIUM_MIN_HEIGHT_PX = MEDIUM_SEARCH_HEIGHT_PX;
static constexpr double LOCAL_MAX_WIDTH_FRAC = 0.46;
static constexpr double LOCAL_MAX_HEIGHT_FRAC = 0.46;
static constexpr double RECOVERY_MAX_WIDTH_FRAC = 0.68;
static constexpr double RECOVERY_MAX_HEIGHT_FRAC = 0.68;

// Association relative to predicted width and height. Growth is intentionally
// more permissive than shrink because a diving aircraft makes the marker grow.
static constexpr double MATCH_MAX_GROW_RATIO = 3.20;
static constexpr double MATCH_MAX_SHRINK_RATIO = 2.20;
static constexpr double MATCH_MAX_ASPECT_CHANGE = 3.00;

// Sticky known-target verifier. Once H is acquired, verify the predicted target
// directly before running generic proposal search. This stays reliable while
// the painted strokes become thick during a fast approach and is cheaper.
static constexpr double TRACK_DIRECT_SHAPE_MIN = 0.66;
static constexpr double TRACK_DIRECT_STRONG_SHAPE = 0.80;
static constexpr double TRACK_DIRECT_PATCH_SCALE = 2.80;
static constexpr double RECOVER_DIRECT_PATCH_SCALE = 4.00;
static constexpr double TRACK_DIRECT_MIN_PATCH = 42.0;
static constexpr int HOLD_LAST_GUIDANCE_FRAMES = 3;
static constexpr int HARD_LOST_SWITCH_MISSES = 7;

// Adaptive contrast scale for generic fallback proposals. Fixed sigma=3 worked
// for tiny H but split a close/thick H into separate edge fragments.
static constexpr double TRACK_RESPONSE_SIGMA_GAIN = 0.14;
static constexpr double TRACK_RESPONSE_SIGMA_MAX = 18.0;
static constexpr double LOCAL_BROAD_RESPONSE_SIGMA = 8.0;

// Switching hysteresis.
static constexpr double ACTIVE_PEAK_DECAY = 0.995;
static constexpr double SWITCH_DROP_FRACTION = 0.40;
static constexpr double SWITCH_MIN_ADVANTAGE = 0.15;
static constexpr int SWITCH_CONFIRM_CHECKS = 3;

// Far-range temporal evidence. The detector keeps a short history of globally
// aligned grayscale frames. Temporal fusion is only an additional FAR-search
// path; strict/near H detection still uses the current frame directly.
static constexpr int TEMPORAL_HISTORY_FRAMES = 5;
static constexpr int TEMPORAL_MIN_FRAMES = 3;
static constexpr std::array<double, TEMPORAL_HISTORY_FRAMES> TEMPORAL_WEIGHTS = {{
    0.45, 0.25, 0.15, 0.10, 0.05
}};
static constexpr double TEMPORAL_ACQUISITION_BONUS = 0.08;

// Revision-6 performance policy. H verification still runs on every processed
// frame, but global scene motion is refreshed every second frame and reuses the
// last per-frame transform in between. Optical flow uses a smaller image and a
// persistent feature set. Far/uncertain acquisition retains the full 5-frame
// aligned history; strong tracking exits early without temporal/global rescans.
static constexpr int MOTION_WORK_MAX_WIDTH = 480;
static constexpr int MOTION_MAX_FEATURES = 120;
static constexpr int MOTION_MIN_REUSE_FEATURES = 60;
static constexpr int MOTION_LK_LEVELS = 2;
static constexpr int MOTION_EVERY_N_FRAMES = 2;
static constexpr int TEMPORAL_ACQUIRE_EVERY_N_FRAMES = 2;
static constexpr int TEMPORAL_VERIFY_EVERY_N_FRAMES = 1;
static constexpr double FAR_SINGLE_FRAME_SHAPE_FLOOR = 0.76;

// Revision-6 false-lock guards. These are acquisition-only identity checks:
// tracking keeps the permissive perspective/size logic once a real H is locked.
// The first guard requires evidence of two separated H stems plus the bridge;
// the second prevents cleanCentralComponents() from manufacturing an H out of
// a cluttered bush/tree/roof by discarding most of the raw threshold structure.
static constexpr double FAR_INTERNAL_CLUTTER_MIN = 0.58;
static constexpr double STRICT_INTERNAL_CLUTTER_MIN = 0.50;

// Revision-6 acquisition guards. FAR topology may be satisfied by one aligned
// 5-frame temporal observation because the two stems / bridge can appear on
// alternating tiny frames. The continuation guard rejects long runway / road /
// roof-edge lines that merely cross like an H but continue well beyond it.
static constexpr int FAR_TOPOLOGY_CONFIRM_MIN = 2;
static constexpr int STRICT_TOPOLOGY_CONFIRM_MIN = 1;
static constexpr int FAR_CLUTTER_CONFIRM_MIN = 2;
static constexpr int STRICT_CLUTTER_CONFIRM_MIN = 1;
static constexpr double EXTERNAL_CONTINUATION_PASS_MIN = 0.48;
static constexpr int FAR_CONTINUATION_CONFIRM_MIN = 2;
static constexpr int STRICT_CONTINUATION_CONFIRM_MIN = 1;
static constexpr double FAR_H_ONLY_OVERRIDE_MIN_CENTER_SCORE = 0.62;
static constexpr int PENDING_FOCUS_MISS_TOLERANCE = 2;
static constexpr int NEW_LOCK_RESET_MISSES = 6;
static constexpr int VERIFIED_LOCK_RESET_MISSES = 30;
static constexpr int VERIFIED_LOCK_FRAMES_FOR_LONG_RECOVERY = 8;

// Revision-9 STAGED tracker remains available as a fallback when global frame
// motion is unavailable. H identity still has to be reverified before TRACK.
static constexpr double RECOVERY_TRACKER_MIN_MAJOR_PX = 18.0;
static constexpr int RECOVERY_TRACKER_RESEED_EVERY_VERIFIED = 3;
static constexpr int RECOVERY_TRACKER_MAX_CONSECUTIVE = 8;
static constexpr int RECOVERY_TRACKER_GUIDANCE_FRAMES = 2;
static constexpr double RECOVERY_TRACKER_MIN_CONF = 0.30;

// Revision-10 FLUENT verified-lock continuation. Once the normal H acquisition
// process has produced a real lock, a temporary H miss propagates the LAST
// VERIFIED target with measured frame motion. This is always reported as
// PREDICTED/RECOVER, never OBSERVED/TRACK. Local H identity is still rechecked
// every frame; MEDIUM/GLOBAL scheduled passes periodically look for recovery or
// a stronger challenger. Prediction expires at the existing verified-lock reset.
static constexpr double FRAME_MOTION_SCALE_MIN = 0.88;
static constexpr double FRAME_MOTION_SCALE_MAX = 1.14;
static constexpr double FRAME_MOTION_MIN_INLIER = 0.42;
static constexpr int FRAME_MOTION_PREDICT_MAX_MISSES = VERIFIED_LOCK_RESET_MISSES;

// Tiny locked H stays in FAR_TRACK policy until it is genuinely resolved.
// This prevents one unusually sharp 10-20 px frame from entering STICKY-FAST.
static constexpr double FAR_TRACK_MAJOR_PX = 32.0;
static constexpr int STRONG_TRACK_CONFIRM_FRAMES = 3;

// CoreLock_04 verified reacquisition escape. The normal sticky association
// remains authoritative until direct H verification has genuinely failed twice.
static constexpr int ESCAPE_MIN_DIRECT_MISSES = 2;
static constexpr int ESCAPE_CONFIRM_CHECKS = 2;

// CoreLock_05 bounded upgrades.
static constexpr int PROVISIONAL_PROMOTE_FRESH = 2;
static constexpr int NESTED_CORE_MAX_CHECKS = 3;

// Optional FC attitude gate. These are camera-attitude values after applying
// the fixed camera mount rotation. If no fresh FC attitude is available the
// gate is ignored and detection runs exactly as before.
static constexpr double CAMERA_PITCH_START_DEG = -15.0;
static constexpr double CAMERA_PITCH_STOP_DEG  = -10.0;

// Canonical verification patch.
static constexpr int CANON = 64;

// ============================================================
// Helpers
// ============================================================

template<typename T>
static T clampv(T value, T low, T high)
{
    return std::max(low, std::min(high, value));
}

static cv::Point2f rectCenter(const cv::Rect2f& r)
{
    return cv::Point2f(
        r.x + 0.5f * r.width,
        r.y + 0.5f * r.height);
}

static double rectMajor(const cv::Rect2f& r)
{
    return std::max(
        static_cast<double>(r.width),
        static_cast<double>(r.height));
}

static double rectMinor(const cv::Rect2f& r)
{
    return std::max(
        1.0,
        static_cast<double>(std::min(r.width, r.height)));
}

static cv::Rect clampRect(const cv::Rect2f& r, const cv::Size& size)
{
    const int x1 = clampv(
        static_cast<int>(std::floor(r.x)),
        0,
        std::max(0, size.width - 1));

    const int y1 = clampv(
        static_cast<int>(std::floor(r.y)),
        0,
        std::max(0, size.height - 1));

    const int x2 = clampv(
        static_cast<int>(std::ceil(r.x + r.width)),
        x1 + 1,
        size.width);

    const int y2 = clampv(
        static_cast<int>(std::ceil(r.y + r.height)),
        y1 + 1,
        size.height);

    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

static cv::Rect centeredRect(
    const cv::Point2f& center,
    double width,
    double height,
    const cv::Size& frameSize)
{
    return clampRect(
        cv::Rect2f(
            static_cast<float>(center.x - 0.5 * width),
            static_cast<float>(center.y - 0.5 * height),
            static_cast<float>(width),
            static_cast<float>(height)),
        frameSize);
}

static cv::Rect expandRect(
    const cv::Rect2f& box,
    const cv::Size& frameSize,
    double factor,
    double minimumSize = 24.0)
{
    const cv::Point2f c = rectCenter(box);
    const double width = std::max(minimumSize, box.width * factor);
    const double height = std::max(minimumSize, box.height * factor);
    return centeredRect(c, width, height, frameSize);
}

static double angleDiff180(double a, double b)
{
    double d = std::fmod(std::abs(a - b), 180.0);
    if (d > 90.0) {
        d = 180.0 - d;
    }
    return d;
}

static double boxDistance(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float ax2 = a.x + a.width;
    const float ay2 = a.y + a.height;
    const float bx2 = b.x + b.width;
    const float by2 = b.y + b.height;

    const double dx = std::max({
        0.0,
        static_cast<double>(a.x - bx2),
        static_cast<double>(b.x - ax2)});

    const double dy = std::max({
        0.0,
        static_cast<double>(a.y - by2),
        static_cast<double>(b.y - ay2)});

    return std::hypot(dx, dy);
}

static cv::Mat makeTextImage(
    const std::string& text,
    int width = 320,
    int height = 180)
{
    cv::Mat image(height, width, CV_8UC1, cv::Scalar(20));
    cv::putText(
        image,
        text,
        cv::Point(18, height / 2),
        cv::FONT_HERSHEY_SIMPLEX,
        0.60,
        cv::Scalar(230),
        2,
        cv::LINE_AA);
    return image;
}

// ============================================================
// Public candidate structures
// ============================================================

} // namespace isolated_h_detail

struct Candidate
{
    cv::RotatedRect ellipseFull;
    // `box` is always the canonical H identity box used by tracking/state.
    // `supportBox` may include nearby pad/ring structure; it is diagnostic/context only.
    cv::Rect2f box;
    cv::Rect2f identityBox;
    cv::Rect2f supportBox;
    cv::Point2f center;

    // Kept compatible with the existing modular loop.
    double ellipseScore = 0.0; // H-footprint + local-clearance quality
    double ringScore = 0.0;    // legacy compatibility; always zero
    double hScore = 0.0;       // H identity quality
    double finalScore = 0.0;
    double ringRadius = 0.0;

    // Revision-6 acquisition diagnostics. These do not tighten an already
    // locked target; they only decide whether SEARCH may commit a new H.
    double acquisitionTopologyScore = 0.0;
    double internalClutterScore = 1.0;
    double externalContinuationScore = 1.0;
    bool acquisitionTopologyPass = false;
    bool acquisitionClutterPass = true;
    bool externalContinuationPass = true;
    bool darkPolarity = false;

    bool hVerified = false;
    bool farMode = false;
    bool strictMode = false;
    bool temporalSupported = false;

    // CoreLock_05 FAR topology diagnostics. Strong H stays on the original
    // fast path; only ambiguous FAR identity needs compatible fresh evidence.
    corelock05_policy::FarSignatureKind farSignatureKind =
        corelock05_policy::FarSignatureKind::Strong;
    bool farMicroAmbiguous = false;
    bool farMicroPass = true;
    double farMicroScore = 1.0;
    double farMicroSideSupport = 1.0;
    double farMicroBridgeSupport = 1.0;
    double farMicroGapEvidence = 1.0;
    double farMicroBlobCompactness = 0.0;
    double farBridgeY = 0.5;
    double farLegTopSupport = 1.0;
    double farLegBottomSupport = 1.0;
    double farLeftTopSupport = 1.0;
    double farLeftBottomSupport = 1.0;
    double farRightTopSupport = 1.0;
    double farRightBottomSupport = 1.0;
    double farGapTop = 1.0;
    double farGapBottom = 1.0;
    double farLeftRightBalance = 1.0;
    double farLargestComponentFraction = 1.0;
    double farTwoLegRowContinuity = 1.0;

    std::string className = "H";
};

static corelock05_policy::FarTopologyMetrics candidateFarTopology(
    const Candidate& c)
{
    corelock05_policy::FarTopologyMetrics m;
    m.leftTop = c.farLeftTopSupport;
    m.leftBottom = c.farLeftBottomSupport;
    m.rightTop = c.farRightTopSupport;
    m.rightBottom = c.farRightBottomSupport;
    m.bridgeStrength = c.farMicroBridgeSupport;
    m.bridgeY = c.farBridgeY;
    m.gapTop = c.farGapTop;
    m.gapBottom = c.farGapBottom;
    m.leftRightBalance = c.farLeftRightBalance;
    m.blobCompactness = c.farMicroBlobCompactness;
    m.largestComponentFraction = c.farLargestComponentFraction;
    m.twoLegRowContinuity = c.farTwoLegRowContinuity;
    return m;
}

struct DetectionOutput
{
    std::vector<Candidate> candidates;
    cv::Mat bestPatch;
    cv::Mat bestHMask;
};

namespace isolated_h_detail {

// ============================================================
// Grayscale H verifier
// ============================================================

class HShapeVerifier
{
public:
    struct Result
    {
        bool valid = false;
        bool farMode = false;
        bool strictMode = false;
        bool darkPolarity = true;
        double shapeScore = 0.0;
        double orientationDeg = 0.0;
        double acquisitionTopologyScore = 0.0;
        double internalClutterScore = 1.0;
        bool acquisitionTopologyPass = false;
        bool acquisitionClutterPass = true;
        cv::Point2f centerInPatch{0.0f, 0.0f};
        cv::Rect foregroundBox;
        cv::Mat normalizedMask;
    };

    Result verify(const cv::Mat& grayPatch, double sourceMajorPx) const
    {
        Result best;

        if (grayPatch.empty() ||
            grayPatch.cols < 5 ||
            grayPatch.rows < 5)
        {
            return best;
        }

        // The previous implementation ran connected components and rotations on
        // the native candidate patch. Near targets can make that patch hundreds
        // of pixels wide. Identity is scale-normalized anyway, so bound all H
        // verification work to a small image and map the center back afterwards.
        const int originalWidth = grayPatch.cols;
        const int originalHeight = grayPatch.rows;
        const int maxDim = std::max(originalWidth, originalHeight);
        const double workScale = maxDim > SHAPE_WORK_MAX_DIM
            ? static_cast<double>(SHAPE_WORK_MAX_DIM) / maxDim
            : 1.0;

        cv::Mat work;
        if (workScale < 0.999) {
            cv::resize(
                grayPatch,
                work,
                cv::Size(),
                workScale,
                workScale,
                cv::INTER_AREA);
        }
        else {
            work = grayPatch;
        }

        cv::Mat smooth;
        cv::GaussianBlur(work, smooth, cv::Size(3, 3), 0.0);

        cv::Mat otsuScratch;
        const double otsu = cv::threshold(
            smooth,
            otsuScratch,
            0,
            255,
            cv::THRESH_BINARY | cv::THRESH_OTSU);

        // Test one threshold for both polarities first. Only ambiguous masks get
        // the two extra threshold variants. This reduces the common path from
        // six masks to two while retaining dark-H and bright-H operation.
        auto evaluateVariant = [&](bool darkPolarity, int offset) {
            const double threshold = clampv(otsu + offset, 1.0, 254.0);

            cv::Mat binary;
            cv::threshold(
                smooth,
                binary,
                threshold,
                255,
                darkPolarity ? cv::THRESH_BINARY_INV
                             : cv::THRESH_BINARY);

            const CleanupStats cleanup = cleanCentralComponents(binary);

            if (cv::countNonZero(binary) < 6) {
                return Result{};
            }

            Result result = evaluateMask(binary, sourceMajorPx);
            result.darkPolarity = darkPolarity;
            result.internalClutterScore = cleanup.clutterScore;
            result.acquisitionClutterPass = result.farMode
                ? cleanup.clutterScore >= FAR_INTERNAL_CLUTTER_MIN
                : cleanup.clutterScore >= STRICT_INTERNAL_CLUTTER_MIN;

            if (result.centerInPatch.x >= 0.0f &&
                result.centerInPatch.y >= 0.0f)
            {
                result.centerInPatch.x = static_cast<float>(
                    result.centerInPatch.x / workScale);
                result.centerInPatch.y = static_cast<float>(
                    result.centerInPatch.y / workScale);
            }

            // Map the measured foreground extent back to the original patch.
            // This lets tracking update width/height from the current H pixels.
            if (workScale < 0.999 && result.foregroundBox.area() > 0) {
                result.foregroundBox.x = static_cast<int>(std::floor(
                    result.foregroundBox.x / workScale));
                result.foregroundBox.y = static_cast<int>(std::floor(
                    result.foregroundBox.y / workScale));
                result.foregroundBox.width = static_cast<int>(std::ceil(
                    result.foregroundBox.width / workScale));
                result.foregroundBox.height = static_cast<int>(std::ceil(
                    result.foregroundBox.height / workScale));
            }

            return result;
        };

        Result dark = evaluateVariant(true, 0);
        Result bright = evaluateVariant(false, 0);
        best = dark.shapeScore >= bright.shapeScore ? dark : bright;

        const bool far = sourceMajorPx < STRICT_MIN_MAJOR_PX;
        const double acceptThreshold = far ? FAR_SHAPE_MIN : STRICT_SHAPE_MIN;

        if (best.shapeScore < acceptThreshold + 0.10) {
            const bool preferredDark = best.darkPolarity;

            for (int offset : {-8, 8}) {
                Result variant = evaluateVariant(preferredDark, offset);
                if (variant.shapeScore > best.shapeScore) {
                    best = std::move(variant);
                }
            }
        }

        best.valid = far
            ? best.shapeScore >= FAR_SHAPE_MIN
            : best.shapeScore >= STRICT_SHAPE_MIN;
        best.farMode = far;
        best.strictMode = !far;
        return best;
    }

private:
    struct CleanupStats
    {
        int eligibleComponents = 0;
        int keptComponents = 0;
        double retainedAreaFraction = 1.0;
        double clutterScore = 1.0;
    };

    static CleanupStats cleanCentralComponents(cv::Mat& binary)
    {
        CleanupStats cleanup;

        if (binary.empty()) {
            cleanup.clutterScore = 0.0;
            return cleanup;
        }

        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;

        const int count = cv::connectedComponentsWithStats(
            binary,
            labels,
            stats,
            centroids,
            8,
            CV_32S);

        struct Component
        {
            int label = 0;
            int area = 0;
            double priority = 0.0;
        };

        const cv::Point2f imageCenter(
            binary.cols * 0.5f,
            binary.rows * 0.5f);

        const double norm = std::max(
            1.0,
            0.5 * std::hypot(
                static_cast<double>(binary.cols),
                static_cast<double>(binary.rows)));

        std::vector<Component> components;
        components.reserve(std::max(0, count - 1));
        int totalEligibleArea = 0;

        for (int label = 1; label < count; ++label) {
            const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(label, cv::CC_STAT_TOP);
            const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);

            const bool touchesBorder =
                x <= 0 || y <= 0 ||
                x + w >= binary.cols ||
                y + h >= binary.rows;

            if (touchesBorder || area < 2) {
                continue;
            }

            const double fill =
                static_cast<double>(area) /
                std::max(1.0, static_cast<double>(w * h));

            // Ignore a large sparse enclosing ring. The H remains central.
            if (w > 0.72 * binary.cols &&
                h > 0.72 * binary.rows &&
                fill < 0.42)
            {
                continue;
            }

            const cv::Point2f center(
                static_cast<float>(centroids.at<double>(label, 0)),
                static_cast<float>(centroids.at<double>(label, 1)));

            const double centerScore = clampv(
                1.0 - cv::norm(center - imageCenter) / norm,
                0.0,
                1.0);

            totalEligibleArea += area;
            components.push_back({
                label,
                area,
                static_cast<double>(area) * (0.42 + 0.58 * centerScore)});
        }

        cleanup.eligibleComponents = static_cast<int>(components.size());

        std::sort(
            components.begin(),
            components.end(),
            [](const Component& a, const Component& b) {
                return a.priority > b.priority;
            });

        if (components.size() > 4) {
            components.resize(4);
        }

        cleanup.keptComponents = static_cast<int>(components.size());
        int keptArea = 0;
        cv::Mat kept = cv::Mat::zeros(binary.size(), CV_8UC1);
        for (const Component& component : components) {
            kept.setTo(255, labels == component.label);
            keptArea += component.area;
        }

        cleanup.retainedAreaFraction = totalEligibleArea > 0
            ? static_cast<double>(keptArea) / totalEligibleArea
            : 0.0;

        // A clean painted H normally consists of one connected component, or a
        // few broken pieces at long range. A bush/tree/roof can contain many
        // central fragments; the old cleanup selected only the best four and
        // accidentally made that clutter look H-like. Score both how much raw
        // structure had to be discarded and how many meaningful pieces existed.
        const double retainedScore = clampv(
            (cleanup.retainedAreaFraction - 0.32) / 0.58,
            0.0,
            1.0);
        const double componentScore = clampv(
            1.0 - std::max(0, cleanup.eligibleComponents - 4) / 8.0,
            0.0,
            1.0);
        cleanup.clutterScore = clampv(
            0.58 * retainedScore + 0.42 * componentScore,
            0.0,
            1.0);

        binary = std::move(kept);
        return cleanup;
    }

    static Result evaluateMask(const cv::Mat& binary, double sourceMajorPx)
    {
        Result best;

        std::vector<cv::Point> points;
        cv::findNonZero(binary, points);

        if (points.size() < 6) {
            return best;
        }

        const cv::Rect foregroundBox = cv::boundingRect(points);
        const cv::RotatedRect rotated = cv::minAreaRect(points);
        const bool far = sourceMajorPx < STRICT_MIN_MAJOR_PX;
        const double acceptThreshold = far ? FAR_SHAPE_MIN : STRICT_SHAPE_MIN;

        auto testAngle = [&](double angle) {
            cv::Mat normalized = rotateCropNormalize(binary, angle);
            if (normalized.empty()) {
                return;
            }

            const double score = far
                ? farProjectionScore(normalized)
                : strictHScore(normalized);
            const AcquisitionTopology topology = acquisitionTopology(normalized);

            if (score > best.shapeScore) {
                best.shapeScore = score;
                best.orientationDeg = angle;
                best.acquisitionTopologyScore = topology.score;
                best.acquisitionTopologyPass = topology.pass;
                best.foregroundBox = foregroundBox;
                best.normalizedMask = normalized.clone();
            }
        };

        // The two bounding-box axes cover all in-plane rotations.
        testAngle(rotated.angle);
        testAngle(rotated.angle + 90.0);

        // Only an ambiguous target needs small angle perturbations.
        if (best.shapeScore < acceptThreshold + 0.10) {
            testAngle(rotated.angle - 10.0);
            testAngle(rotated.angle + 10.0);
            testAngle(rotated.angle + 80.0);
            testAngle(rotated.angle + 100.0);
        }

        best.valid = far
            ? best.shapeScore >= FAR_SHAPE_MIN
            : best.shapeScore >= STRICT_SHAPE_MIN;
        best.farMode = far;
        best.strictMode = !far;

        const cv::Moments moments = cv::moments(binary, true);
        if (std::abs(moments.m00) > 1e-6) {
            best.centerInPatch = cv::Point2f(
                static_cast<float>(moments.m10 / moments.m00),
                static_cast<float>(moments.m01 / moments.m00));
        }
        else {
            best.centerInPatch = cv::Point2f(
                foregroundBox.x + 0.5f * foregroundBox.width,
                foregroundBox.y + 0.5f * foregroundBox.height);
        }

        return best;
    }

    static cv::Mat rotateCropNormalize(
        const cv::Mat& binary,
        double angleDeg)
    {
        const cv::Point2f center(
            binary.cols * 0.5f,
            binary.rows * 0.5f);

        const cv::Mat transform = cv::getRotationMatrix2D(
            center,
            angleDeg,
            1.0);

        cv::Mat rotated;
        cv::warpAffine(
            binary,
            rotated,
            transform,
            binary.size(),
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0));

        std::vector<cv::Point> points;
        cv::findNonZero(rotated, points);
        if (points.empty()) {
            return {};
        }

        cv::Rect box = cv::boundingRect(points);
        if (box.width < 2 || box.height < 2) {
            return {};
        }

        const int pad = std::max(
            2,
            static_cast<int>(0.08 * std::max(box.width, box.height)));

        box.x = std::max(0, box.x - pad);
        box.y = std::max(0, box.y - pad);
        box.width = std::min(rotated.cols - box.x, box.width + 2 * pad);
        box.height = std::min(rotated.rows - box.y, box.height + 2 * pad);

        cv::Mat normalized;
        cv::resize(
            rotated(box),
            normalized,
            cv::Size(CANON, CANON),
            0,
            0,
            cv::INTER_NEAREST);

        return normalized;
    }

    static double zoneMean(
        const cv::Mat& binary,
        double x1,
        double y1,
        double x2,
        double y2)
    {
        const int ix1 = clampv(
            static_cast<int>(std::lround(x1 * CANON)),
            0,
            CANON - 1);

        const int iy1 = clampv(
            static_cast<int>(std::lround(y1 * CANON)),
            0,
            CANON - 1);

        const int ix2 = clampv(
            static_cast<int>(std::lround(x2 * CANON)),
            ix1 + 1,
            CANON);

        const int iy2 = clampv(
            static_cast<int>(std::lround(y2 * CANON)),
            iy1 + 1,
            CANON);

        return cv::mean(
            binary(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1)))[0] /
            255.0;
    }

    static double rowCoverage(
        const cv::Mat& binary,
        double x1,
        double x2,
        double threshold)
    {
        const int ix1 = clampv(
            static_cast<int>(x1 * CANON),
            0,
            CANON - 1);

        const int ix2 = clampv(
            static_cast<int>(x2 * CANON),
            ix1 + 1,
            CANON);

        int supported = 0;
        int total = 0;

        for (int y = static_cast<int>(0.06 * CANON);
             y < static_cast<int>(0.94 * CANON);
             ++y)
        {
            const double mean = cv::mean(
                binary(cv::Rect(ix1, y, ix2 - ix1, 1)))[0] / 255.0;

            if (mean >= threshold) {
                ++supported;
            }
            ++total;
        }

        return total > 0
            ? static_cast<double>(supported) / total
            : 0.0;
    }

    static double columnCoverage(
        const cv::Mat& binary,
        double y1,
        double y2,
        double threshold)
    {
        const int iy1 = clampv(
            static_cast<int>(y1 * CANON),
            0,
            CANON - 1);

        const int iy2 = clampv(
            static_cast<int>(y2 * CANON),
            iy1 + 1,
            CANON);

        int supported = 0;
        int total = 0;

        for (int x = static_cast<int>(0.18 * CANON);
             x < static_cast<int>(0.82 * CANON);
             ++x)
        {
            const double mean = cv::mean(
                binary(cv::Rect(x, iy1, 1, iy2 - iy1)))[0] / 255.0;

            if (mean >= threshold) {
                ++supported;
            }
            ++total;
        }

        return total > 0
            ? static_cast<double>(supported) / total
            : 0.0;
    }

    static double projectionGeometryScore(const cv::Mat& binary)
    {
        const double left = zoneMean(binary, 0.10, 0.06, 0.39, 0.94);
        const double right = zoneMean(binary, 0.61, 0.06, 0.90, 0.94);
        const double bridge = zoneMean(binary, 0.26, 0.39, 0.74, 0.61);

        const double topGap = zoneMean(binary, 0.42, 0.06, 0.58, 0.34);
        const double bottomGap = zoneMean(binary, 0.42, 0.66, 0.58, 0.94);

        const double sideBalance = clampv(
            1.0 - std::abs(left - right) / 0.30,
            0.0,
            1.0);

        const double leftScore = clampv((left - 0.10) / 0.34, 0.0, 1.0);
        const double rightScore = clampv((right - 0.10) / 0.34, 0.0, 1.0);
        const double bridgeScore = clampv((bridge - 0.08) / 0.34, 0.0, 1.0);
        const double gapScore = clampv(
            1.0 - (topGap + bottomGap) / 0.62,
            0.0,
            1.0);

        const double leftCoverage = rowCoverage(binary, 0.10, 0.39, 0.08);
        const double rightCoverage = rowCoverage(binary, 0.61, 0.90, 0.08);
        const double bridgeCoverage = columnCoverage(binary, 0.39, 0.61, 0.08);

        const double continuity =
            0.40 * std::min(leftCoverage, rightCoverage) +
            0.25 * std::max(leftCoverage, rightCoverage) +
            0.35 * bridgeCoverage;

        return clampv(
            0.20 * leftScore +
            0.20 * rightScore +
            0.24 * bridgeScore +
            0.13 * gapScore +
            0.10 * sideBalance +
            0.13 * continuity,
            0.0,
            1.0);
    }

    static const std::vector<cv::Mat>& expectedMasks()
    {
        static const std::vector<cv::Mat> masks = []() {
            std::vector<cv::Mat> result;

            for (double widthFraction : {0.10, 0.14, 0.18}) {
                cv::Mat expected = cv::Mat::zeros(CANON, CANON, CV_8UC1);
                const int bar = std::max(
                    2,
                    static_cast<int>(widthFraction * CANON));
                const int xLeft = static_cast<int>(0.24 * CANON);
                const int xRight = static_cast<int>(0.76 * CANON);
                const int yTop = static_cast<int>(0.08 * CANON);
                const int yBottom = static_cast<int>(0.92 * CANON);
                const int yMiddle = static_cast<int>(0.50 * CANON);

                cv::line(expected, cv::Point(xLeft, yTop),
                         cv::Point(xLeft, yBottom), cv::Scalar(255), bar, cv::LINE_8);
                cv::line(expected, cv::Point(xRight, yTop),
                         cv::Point(xRight, yBottom), cv::Scalar(255), bar, cv::LINE_8);
                cv::line(expected, cv::Point(xLeft, yMiddle),
                         cv::Point(xRight, yMiddle), cv::Scalar(255), bar, cv::LINE_8);

                result.push_back(std::move(expected));
            }

            return result;
        }();

        return masks;
    }

    static double expectedMaskF1(const cv::Mat& binary)
    {
        double best = 0.0;
        const int observedCount = cv::countNonZero(binary);

        if (observedCount <= 0) {
            return 0.0;
        }

        for (const cv::Mat& expected : expectedMasks()) {
            cv::Mat intersection;
            cv::bitwise_and(binary, expected, intersection);

            const double inter = cv::countNonZero(intersection);
            const double expectedCount = cv::countNonZero(expected);
            const double precision = inter /
                std::max(1.0, static_cast<double>(observedCount));
            const double recall = inter / std::max(1.0, expectedCount);
            const double f1 = 2.0 * precision * recall /
                std::max(1e-6, precision + recall);

            best = std::max(best, f1);
        }

        return best;
    }

    struct AcquisitionTopology
    {
        double score = 0.0;
        bool pass = false;
    };

    static AcquisitionTopology acquisitionTopology(const cv::Mat& binary)
    {
        // Explicit H topology for NEW acquisition only. The normal shape score
        // is intentionally perspective tolerant; this second check prevents a
        // plus sign, T, single line, bush branches, or two unrelated stems from
        // becoming a committed target. One of the four stem quadrants may be
        // weak/missing, which preserves the earlier multi-frame partial-H design.
        const double leftTop = zoneMean(binary, 0.12, 0.08, 0.36, 0.40);
        const double leftBottom = zoneMean(binary, 0.12, 0.60, 0.36, 0.92);
        const double rightTop = zoneMean(binary, 0.64, 0.08, 0.88, 0.40);
        const double rightBottom = zoneMean(binary, 0.64, 0.60, 0.88, 0.92);
        const double bridge = zoneMean(binary, 0.40, 0.42, 0.60, 0.58);
        const double topGap = zoneMean(binary, 0.43, 0.08, 0.57, 0.35);
        const double bottomGap = zoneMean(binary, 0.43, 0.65, 0.57, 0.92);

        auto stemSupport = [](double value) {
            return clampv((value - 0.04) / 0.20, 0.0, 1.0);
        };

        std::array<double, 4> stems = {{
            stemSupport(leftTop),
            stemSupport(leftBottom),
            stemSupport(rightTop),
            stemSupport(rightBottom)
        }};
        std::sort(stems.begin(), stems.end());

        // stems[1] means at least three of four stem quadrants have support.
        const double threeOfFour = stems[1];
        const double bothSides = std::min(
            std::max(stemSupport(leftTop), stemSupport(leftBottom)),
            std::max(stemSupport(rightTop), stemSupport(rightBottom)));
        const double bridgeScore = clampv(
            (bridge - 0.05) / 0.30, 0.0, 1.0);
        const double gapScore = clampv(
            1.0 - (topGap + bottomGap) / 0.50, 0.0, 1.0);

        AcquisitionTopology out;
        out.score = clampv(
            0.35 * threeOfFour +
            0.20 * bothSides +
            0.30 * bridgeScore +
            0.15 * gapScore,
            0.0,
            1.0);

        out.pass =
            threeOfFour >= 0.22 &&
            bothSides >= 0.25 &&
            bridgeScore >= 0.25 &&
            gapScore >= 0.10;
        return out;
    }

    static double farProjectionScore(const cv::Mat& binary)
    {
        const double geometry = projectionGeometryScore(binary);
        const double f1 = expectedMaskF1(binary);

        return clampv(
            0.50 * geometry +
            0.50 * f1,
            0.0,
            1.0);
    }

    static double strictHScore(const cv::Mat& binary)
    {
        const double geometry = projectionGeometryScore(binary);
        const double f1 = expectedMaskF1(binary);

        return clampv(
            0.40 * geometry +
            0.60 * f1,
            0.0,
            1.0);
    }
};

// ============================================================
// Pure-H resolved integrity guard
// ============================================================

class HIntegrityVerifier
{
public:
    struct Result
    {
        bool pass = true;
        bool hardChecked = false;
        double score = 1.0;
        double spanningComponentFraction = 0.0;
        double threeOfFourStemSupport = 0.0;
        double bothSidesSupport = 0.0;
        double bridgeSupport = 0.0;
        double centerGapScore = 0.0;
        double fillRatio = 0.0;
        double legRowContinuity = 0.0;
        double bridgeRowContinuity = 0.0;
        bool bridgeConnectivity = false;
    };

    Result evaluate(const cv::Mat& normalizedHMask, double sourceMajorPx) const
    {
        Result result;

        if (sourceMajorPx < STRICT_MIN_MAJOR_PX) {
            result.pass = true;
            result.hardChecked = false;
            return result;
        }

        result.hardChecked = true;
        if (normalizedHMask.empty()) {
            result.pass = false;
            result.score = 0.0;
            return result;
        }

        cv::Mat mask;
        if (normalizedHMask.size() != cv::Size(CANON, CANON)) {
            cv::resize(
                normalizedHMask,
                mask,
                cv::Size(CANON, CANON),
                0,
                0,
                cv::INTER_NEAREST);
        }
        else {
            mask = normalizedHMask.clone();
        }
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

        std::vector<cv::Point> foreground;
        cv::findNonZero(mask, foreground);
        if (foreground.size() < 6) {
            result.pass = false;
            result.score = 0.0;
            return result;
        }

        const cv::Rect box = cv::boundingRect(foreground);
        result.fillRatio = static_cast<double>(foreground.size()) /
            std::max(1.0, static_cast<double>(box.area()));

        const double leftTop = zoneMean(mask, 0.12, 0.08, 0.36, 0.40);
        const double leftBottom = zoneMean(mask, 0.12, 0.60, 0.36, 0.92);
        const double rightTop = zoneMean(mask, 0.64, 0.08, 0.88, 0.40);
        const double rightBottom = zoneMean(mask, 0.64, 0.60, 0.88, 0.92);
        const double bridge = zoneMean(mask, 0.34, 0.40, 0.66, 0.60);
        const double topGap = zoneMean(mask, 0.43, 0.08, 0.57, 0.34);
        const double bottomGap = zoneMean(mask, 0.43, 0.66, 0.57, 0.92);

        auto support = [](double value) {
            return clampv((value - 0.035) / 0.22, 0.0, 1.0);
        };

        std::array<double, 4> stems = {{
            support(leftTop),
            support(leftBottom),
            support(rightTop),
            support(rightBottom)
        }};
        std::sort(stems.begin(), stems.end());
        result.threeOfFourStemSupport = stems[1];
        result.bothSidesSupport = std::min(
            std::max(support(leftTop), support(leftBottom)),
            std::max(support(rightTop), support(rightBottom)));
        result.bridgeSupport = clampv((bridge - 0.035) / 0.27, 0.0, 1.0);
        result.centerGapScore = clampv(
            1.0 - (topGap + bottomGap) / 0.58,
            0.0,
            1.0);

        // Three-stroke run profile. After orientation normalization, a resolved
        // H should show two separated leg runs over the upper/lower spans and a
        // real horizontal bridge over several middle rows. This is a negative
        // guard against AI5-style curved/blob clutter, not a new H detector.
        cv::Mat profileMask;
        cv::morphologyEx(
            mask,
            profileMask,
            cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        auto twoLegRowFraction = [&](double y1, double y2) {
            const int ys = clampv(static_cast<int>(y1 * CANON), 0, CANON - 1);
            const int ye = clampv(static_cast<int>(y2 * CANON), ys + 1, CANON);
            int good = 0;
            int totalRows = 0;
            for (int y = ys; y < ye; ++y) {
                const cv::Mat row = profileMask.row(y);
                const int left = cv::countNonZero(row.colRange(
                    static_cast<int>(0.08 * CANON),
                    static_cast<int>(0.42 * CANON)));
                const int right = cv::countNonZero(row.colRange(
                    static_cast<int>(0.58 * CANON),
                    static_cast<int>(0.92 * CANON)));
                const int centerPixels = cv::countNonZero(row.colRange(
                    static_cast<int>(0.43 * CANON),
                    static_cast<int>(0.57 * CANON)));
                if (left >= 2 && right >= 2 && centerPixels <= 2) {
                    ++good;
                }
                ++totalRows;
            }
            return totalRows > 0
                ? static_cast<double>(good) / totalRows
                : 0.0;
        };

        const double topLegRows = twoLegRowFraction(0.08, 0.38);
        const double bottomLegRows = twoLegRowFraction(0.62, 0.92);
        result.legRowContinuity = std::min(topLegRows, bottomLegRows);

        cv::Mat bridgeProfile;
        cv::morphologyEx(
            profileMask,
            bridgeProfile,
            cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1)));
        int bridgeRowsGood = 0;
        int bridgeRowsTotal = 0;
        for (int y = static_cast<int>(0.36 * CANON);
             y < static_cast<int>(0.64 * CANON); ++y)
        {
            const cv::Mat row = bridgeProfile.row(y);
            std::vector<cv::Point> rowPoints;
            cv::findNonZero(row, rowPoints);
            bool spans = false;
            if (!rowPoints.empty()) {
                int minX = CANON;
                int maxX = -1;
                for (const cv::Point& point : rowPoints) {
                    minX = std::min(minX, point.x);
                    maxX = std::max(maxX, point.x);
                }
                const int centerSupport = cv::countNonZero(row.colRange(
                    static_cast<int>(0.43 * CANON),
                    static_cast<int>(0.57 * CANON)));
                spans =
                    minX < static_cast<int>(0.35 * CANON) &&
                    maxX > static_cast<int>(0.65 * CANON) &&
                    centerSupport >= 2;
            }
            if (spans) {
                ++bridgeRowsGood;
            }
            ++bridgeRowsTotal;
        }
        result.bridgeRowContinuity = bridgeRowsTotal > 0
            ? static_cast<double>(bridgeRowsGood) / bridgeRowsTotal
            : 0.0;

        // A real resolved H should contain one connected structure that reaches
        // both legs and the bridge. A small close operation tolerates threshold
        // cracks without joining widely separated AI5-style blobs.
        cv::Mat connectedMask;
        cv::morphologyEx(
            mask,
            connectedMask,
            cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        cv::Mat labels, stats, centroids;
        const int count = cv::connectedComponentsWithStats(
            connectedMask,
            labels,
            stats,
            centroids,
            8,
            CV_32S);
        const int total = std::max(1, cv::countNonZero(connectedMask));

        int largest = 0;
        bool bridgeConnected = false;
        for (int label = 1; label < count; ++label) {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            largest = std::max(largest, area);

            const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(label, cv::CC_STAT_TOP);
            const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
            const double fraction = static_cast<double>(area) / total;

            if (fraction < 0.42) {
                continue;
            }

            cv::Mat component = labels == label;
            const int leftSupport = cv::countNonZero(component(cv::Rect(
                static_cast<int>(0.08 * CANON),
                static_cast<int>(0.12 * CANON),
                static_cast<int>(0.35 * CANON),
                static_cast<int>(0.76 * CANON))));
            const int rightSupport = cv::countNonZero(component(cv::Rect(
                static_cast<int>(0.57 * CANON),
                static_cast<int>(0.12 * CANON),
                static_cast<int>(0.35 * CANON),
                static_cast<int>(0.76 * CANON))));
            const int bridgeSupport = cv::countNonZero(component(cv::Rect(
                static_cast<int>(0.30 * CANON),
                static_cast<int>(0.36 * CANON),
                static_cast<int>(0.40 * CANON),
                static_cast<int>(0.28 * CANON))));

            const bool spansHorizontally =
                x < static_cast<int>(0.43 * CANON) &&
                x + w > static_cast<int>(0.57 * CANON);
            const bool hasVerticalExtent = h > static_cast<int>(0.42 * CANON);

            if (spansHorizontally && hasVerticalExtent &&
                leftSupport >= 3 && rightSupport >= 3 && bridgeSupport >= 3)
            {
                bridgeConnected = true;
            }
        }

        result.spanningComponentFraction = static_cast<double>(largest) / total;
        result.bridgeConnectivity = bridgeConnected;

        pure_h_corelock::IntegrityMetrics metrics;
        metrics.resolved = true;
        metrics.spanningComponentFraction = result.spanningComponentFraction;
        metrics.threeOfFourStemSupport = result.threeOfFourStemSupport;
        metrics.bothSidesSupport = result.bothSidesSupport;
        metrics.bridgeSupport = result.bridgeSupport;
        metrics.centerGapScore = result.centerGapScore;
        metrics.fillRatio = result.fillRatio;
        metrics.legRowContinuity = result.legRowContinuity;
        metrics.bridgeRowContinuity = result.bridgeRowContinuity;
        metrics.bridgeConnectivity = result.bridgeConnectivity;

        const auto policy = pure_h_corelock::evaluateIntegrity(metrics);
        result.pass = policy.pass;
        result.hardChecked = policy.hardChecked;
        result.score = policy.score;
        return result;
    }

private:
    static double zoneMean(
        const cv::Mat& binary,
        double x1,
        double y1,
        double x2,
        double y2)
    {
        const int ix1 = clampv(
            static_cast<int>(std::lround(x1 * CANON)), 0, CANON - 1);
        const int iy1 = clampv(
            static_cast<int>(std::lround(y1 * CANON)), 0, CANON - 1);
        const int ix2 = clampv(
            static_cast<int>(std::lround(x2 * CANON)), ix1 + 1, CANON);
        const int iy2 = clampv(
            static_cast<int>(std::lround(y2 * CANON)), iy1 + 1, CANON);
        return cv::mean(binary(cv::Rect(
            ix1, iy1, ix2 - ix1, iy2 - iy1)))[0] / 255.0;
    }
};

// ============================================================
// FAR micro-structure ambiguity guard + resolved support analysis
// ============================================================

class FarMicroHVerifier
{
public:
    struct Result
    {
        bool pass = true;
        bool ambiguous = false;
        double score = 1.0;
        corelock04_policy::FarMicroMetrics metrics;
        corelock05_policy::FarTopologyMetrics topology;
        corelock05_policy::FarSignatureKind signatureKind =
            corelock05_policy::FarSignatureKind::Strong;
    };

    Result evaluate(const cv::Mat& normalizedMask) const
    {
        Result out;
        if (normalizedMask.empty()) {
            out.pass = false;
            out.ambiguous = true;
            out.score = 0.0;
            out.signatureKind = corelock05_policy::FarSignatureKind::Reject;
            return out;
        }

        cv::Mat mask;
        if (normalizedMask.size() != cv::Size(CANON, CANON)) {
            cv::resize(normalizedMask, mask, cv::Size(CANON, CANON), 0, 0, cv::INTER_NEAREST);
        }
        else {
            mask = normalizedMask.clone();
        }
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

        std::vector<cv::Point> points;
        cv::findNonZero(mask, points);
        if (points.size() < 5) {
            out.pass = false;
            out.ambiguous = true;
            out.score = 0.0;
            out.signatureKind = corelock05_policy::FarSignatureKind::Reject;
            return out;
        }

        auto zoneMean = [&](double x1, double y1, double x2, double y2) {
            x1 = clampv(x1, 0.0, 0.98);
            y1 = clampv(y1, 0.0, 0.98);
            x2 = clampv(x2, x1 + 0.01, 1.0);
            y2 = clampv(y2, y1 + 0.01, 1.0);
            const int ix1 = clampv(static_cast<int>(std::floor(x1 * CANON)), 0, CANON - 1);
            const int iy1 = clampv(static_cast<int>(std::floor(y1 * CANON)), 0, CANON - 1);
            const int ix2 = clampv(static_cast<int>(std::ceil(x2 * CANON)), ix1 + 1, CANON);
            const int iy2 = clampv(static_cast<int>(std::ceil(y2 * CANON)), iy1 + 1, CANON);
            return cv::mean(mask(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1)))[0] / 255.0;
        };

        auto stemSupport = [](double v) {
            return clampv((v - 0.014) / 0.155, 0.0, 1.0);
        };

        double bestBridgeRaw = -1.0;
        double bestBridgeY = 0.5;
        for (int yi = 14; yi <= 50; ++yi) {
            const double y = static_cast<double>(yi) / CANON;
            const double stripe = zoneMean(0.20, y - 0.035, 0.80, y + 0.035);
            const double center = zoneMean(0.40, y - 0.045, 0.60, y + 0.045);
            const double sideL = zoneMean(0.08, y - 0.050, 0.40, y + 0.050);
            const double sideR = zoneMean(0.60, y - 0.050, 0.92, y + 0.050);
            const double raw = 0.50 * center + 0.30 * stripe + 0.10 * sideL + 0.10 * sideR;
            if (raw > bestBridgeRaw) {
                bestBridgeRaw = raw;
                bestBridgeY = y;
            }
        }

        const double topEnd = std::max(0.15, bestBridgeY - 0.08);
        const double bottomStart = std::min(0.85, bestBridgeY + 0.08);

        const double leftTop = stemSupport(zoneMean(0.08, 0.06, 0.40, topEnd));
        const double rightTop = stemSupport(zoneMean(0.60, 0.06, 0.92, topEnd));
        const double leftBottom = stemSupport(zoneMean(0.08, bottomStart, 0.40, 0.94));
        const double rightBottom = stemSupport(zoneMean(0.60, bottomStart, 0.92, 0.94));
        const double bridge = clampv((bestBridgeRaw - 0.025) / 0.30, 0.0, 1.0);

        const double topGapOcc = zoneMean(0.43, 0.06, 0.57, topEnd);
        const double bottomGapOcc = zoneMean(0.43, bottomStart, 0.57, 0.94);
        const double gapTop = clampv(1.0 - topGapOcc / 0.36, 0.0, 1.0);
        const double gapBottom = clampv(1.0 - bottomGapOcc / 0.36, 0.0, 1.0);

        const double leftAvg = 0.5 * (leftTop + leftBottom);
        const double rightAvg = 0.5 * (rightTop + rightBottom);
        const double leftRightBalance = clampv(
            1.0 - std::abs(leftAvg - rightAvg) /
                std::max(0.18, std::max(leftAvg, rightAvg)),
            0.0, 1.0);

        const cv::Rect box = cv::boundingRect(points);
        const double fill = static_cast<double>(points.size()) /
            std::max(1.0, static_cast<double>(box.area()));

        cv::Mat labels, stats, centroids;
        const int componentCount = cv::connectedComponentsWithStats(
            mask, labels, stats, centroids, 8, CV_32S);
        int largestComponent = 0;
        for (int label = 1; label < componentCount; ++label) {
            largestComponent = std::max(
                largestComponent,
                stats.at<int>(label, cv::CC_STAT_AREA));
        }
        const double largestFraction = static_cast<double>(largestComponent) /
            std::max(1.0, static_cast<double>(points.size()));

        int twoSideRows = 0;
        int testedRows = 0;
        for (int y = static_cast<int>(0.08 * CANON);
             y < static_cast<int>(0.92 * CANON); ++y)
        {
            const cv::Mat row = mask.row(y);
            const int leftPixels = cv::countNonZero(row.colRange(
                static_cast<int>(0.08 * CANON),
                static_cast<int>(0.42 * CANON)));
            const int rightPixels = cv::countNonZero(row.colRange(
                static_cast<int>(0.58 * CANON),
                static_cast<int>(0.92 * CANON)));
            const int centerPixels = cv::countNonZero(row.colRange(
                static_cast<int>(0.43 * CANON),
                static_cast<int>(0.57 * CANON)));
            if (leftPixels >= 2 && rightPixels >= 2 && centerPixels <= 3) {
                ++twoSideRows;
            }
            ++testedRows;
        }

        out.topology.leftTop = leftTop;
        out.topology.leftBottom = leftBottom;
        out.topology.rightTop = rightTop;
        out.topology.rightBottom = rightBottom;
        out.topology.bridgeStrength = bridge;
        out.topology.bridgeY = bestBridgeY;
        out.topology.gapTop = gapTop;
        out.topology.gapBottom = gapBottom;
        out.topology.leftRightBalance = leftRightBalance;
        out.topology.blobCompactness = clampv(fill, 0.0, 1.0);
        out.topology.largestComponentFraction = clampv(largestFraction, 0.0, 1.0);
        out.topology.twoLegRowContinuity = testedRows > 0
            ? static_cast<double>(twoSideRows) / testedRows
            : 0.0;

        out.metrics.sideSupport = std::min(
            0.5 * (leftTop + leftBottom),
            0.5 * (rightTop + rightBottom));
        out.metrics.bridgeSupport = bridge;
        out.metrics.gapEvidence = std::min(gapTop, gapBottom);
        out.metrics.blobCompactness = out.topology.blobCompactness;
        out.metrics.largestComponentFraction = out.topology.largestComponentFraction;
        out.metrics.twoSideRowContinuity = testedRows > 0
            ? static_cast<double>(twoSideRows) / testedRows
            : 0.0;

        const auto decision = corelock05_policy::classifyFarTopology(out.topology);
        out.signatureKind = decision.kind;
        out.pass = decision.kind != corelock05_policy::FarSignatureKind::Reject;
        out.ambiguous = decision.kind == corelock05_policy::FarSignatureKind::Ambiguous;
        out.score = decision.score;
        return out;
    }
};

class HIdentitySupportAnalyzer
{
public:
    struct Result
    {
        bool supportHeavy = false;
        double unexplainedFraction = 0.0;
        double centerGapContamination = 0.0;
    };

    Result evaluate(const cv::Mat& normalizedMask) const
    {
        Result out;
        if (normalizedMask.empty()) return out;

        cv::Mat mask;
        if (normalizedMask.size() != cv::Size(CANON, CANON)) {
            cv::resize(normalizedMask, mask, cv::Size(CANON, CANON), 0, 0, cv::INTER_NEAREST);
        }
        else {
            mask = normalizedMask.clone();
        }
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

        const int total = cv::countNonZero(mask);
        if (total <= 0) return out;

        // Broad expected-H support zones. Perspective/shear remain inside these
        // regions, while enclosing rings/squares create substantial unexplained
        // foreground outside them.
        cv::Mat allowed = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::rectangle(allowed,
            cv::Rect(static_cast<int>(0.05 * CANON), static_cast<int>(0.03 * CANON),
                     static_cast<int>(0.40 * CANON), static_cast<int>(0.94 * CANON)),
            cv::Scalar(255), cv::FILLED);
        cv::rectangle(allowed,
            cv::Rect(static_cast<int>(0.55 * CANON), static_cast<int>(0.03 * CANON),
                     static_cast<int>(0.40 * CANON), static_cast<int>(0.94 * CANON)),
            cv::Scalar(255), cv::FILLED);
        cv::rectangle(allowed,
            cv::Rect(static_cast<int>(0.18 * CANON), static_cast<int>(0.31 * CANON),
                     static_cast<int>(0.64 * CANON), static_cast<int>(0.38 * CANON)),
            cv::Scalar(255), cv::FILLED);

        cv::Mat outsideAllowed;
        cv::bitwise_not(allowed, outsideAllowed);
        cv::bitwise_and(mask, outsideAllowed, outsideAllowed);
        out.unexplainedFraction = static_cast<double>(cv::countNonZero(outsideAllowed)) / total;

        const cv::Rect topGap(
            static_cast<int>(0.43 * CANON), static_cast<int>(0.04 * CANON),
            static_cast<int>(0.14 * CANON), static_cast<int>(0.30 * CANON));
        const cv::Rect bottomGap(
            static_cast<int>(0.43 * CANON), static_cast<int>(0.66 * CANON),
            static_cast<int>(0.14 * CANON), static_cast<int>(0.30 * CANON));
        const int gapPixels = cv::countNonZero(mask(topGap)) + cv::countNonZero(mask(bottomGap));
        out.centerGapContamination = static_cast<double>(gapPixels) /
            std::max(1.0, static_cast<double>(topGap.area() + bottomGap.area()));

        out.supportHeavy =
            out.unexplainedFraction > 0.20 ||
            out.centerGapContamination > 0.22;
        return out;
    }
};

// ============================================================
// H-core refinement for proposals that contain surrounding geometry
// ============================================================

class HCoreRefiner
{
public:
    struct Core
    {
        cv::Rect box;
        double major = 0.0;
        double priority = 0.0;
    };

    std::vector<Core> refine(
        const cv::Mat& grayPatch,
        const HShapeVerifier::Result& seedShape,
        double proposalMajor) const
    {
        std::vector<Core> cores;
        if (grayPatch.empty() || proposalMajor < STRICT_MIN_MAJOR_PX) {
            return cores;
        }

        auto addCore = [&](const cv::Rect& rawBox, double priority) {
            cv::Rect box = rawBox & cv::Rect(0, 0, grayPatch.cols, grayPatch.rows);
            if (box.width < 3 || box.height < 3) {
                return;
            }
            const double major = std::max(box.width, box.height);
            const double minor = std::min(box.width, box.height);
            if (major < FAR_MIN_MAJOR_PX || minor < FAR_MIN_MINOR_PX) {
                return;
            }
            // Refinement exists to remove enclosing geometry. A box almost as
            // large as the original proposal is not a useful H core.
            if (major > 0.90 * proposalMajor) {
                return;
            }

            for (const Core& existing : cores) {
                const cv::Rect intersection = existing.box & box;
                const double inter = intersection.area();
                const double uni = existing.box.area() + box.area() - inter;
                if (uni > 0.0 && inter / uni > 0.58) {
                    return;
                }
            }
            cores.push_back({box, major, priority});
        };

        // First candidate: reuse what Rev11's own component cleanup considered
        // foreground. This is cheap and often already isolates the H from a
        // surrounding sparse ring/square even when STRICT mode rejected it.
        if (seedShape.foregroundBox.area() > 0) {
            const cv::Rect seed = seedShape.foregroundBox &
                cv::Rect(0, 0, grayPatch.cols, grayPatch.rows);
            const cv::Point2f c(
                seed.x + 0.5f * seed.width,
                seed.y + 0.5f * seed.height);
            const cv::Point2f pc(
                0.5f * grayPatch.cols,
                0.5f * grayPatch.rows);
            const double norm = std::max(1.0, 0.5 * std::hypot(
                static_cast<double>(grayPatch.cols),
                static_cast<double>(grayPatch.rows)));
            const double centerScore = clampv(
                1.0 - cv::norm(c - pc) / norm, 0.0, 1.0);
            addCore(seed, 1.20 + 0.40 * seedShape.shapeScore + 0.20 * centerScore);
        }

        const int maxDim = std::max(grayPatch.cols, grayPatch.rows);
        const double scale = maxDim > CORE_REFINE_WORK_MAX_DIM
            ? static_cast<double>(CORE_REFINE_WORK_MAX_DIM) / maxDim
            : 1.0;
        cv::Mat work;
        if (scale < 0.999) {
            cv::resize(grayPatch, work, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        else {
            work = grayPatch;
        }

        cv::Mat smooth;
        cv::GaussianBlur(work, smooth, cv::Size(3, 3), 0.0);
        cv::Mat scratch;
        const double otsu = cv::threshold(
            smooth, scratch, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        for (bool darkPolarity : {true, false}) {
            cv::Mat binary;
            cv::threshold(
                smooth,
                binary,
                otsu,
                255,
                darkPolarity ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);

            cv::Mat labels, stats, centroids;
            const int count = cv::connectedComponentsWithStats(
                binary, labels, stats, centroids, 8, CV_32S);
            const cv::Point2f imageCenter(
                0.5f * work.cols, 0.5f * work.rows);
            const double norm = std::max(1.0, 0.5 * std::hypot(
                static_cast<double>(work.cols),
                static_cast<double>(work.rows)));

            for (int label = 1; label < count; ++label) {
                const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
                const int y = stats.at<int>(label, cv::CC_STAT_TOP);
                const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
                const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
                const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                if (area < 5 || w < 3 || h < 3) {
                    continue;
                }

                const bool touchesBorder =
                    x <= 0 || y <= 0 ||
                    x + w >= work.cols || y + h >= work.rows;
                if (touchesBorder) {
                    continue;
                }

                const double fill = static_cast<double>(area) /
                    std::max(1.0, static_cast<double>(w * h));
                // Surrounding circle/square tends to be a large sparse component.
                if (w > 0.68 * work.cols && h > 0.68 * work.rows && fill < 0.56) {
                    continue;
                }

                const cv::Point2f center(
                    static_cast<float>(centroids.at<double>(label, 0)),
                    static_cast<float>(centroids.at<double>(label, 1)));
                const double centerScore = clampv(
                    1.0 - cv::norm(center - imageCenter) / norm,
                    0.0,
                    1.0);
                const double aspect = static_cast<double>(w) / std::max(1, h);
                if (aspect < 0.22 || aspect > 4.5) {
                    continue;
                }
                const double fillScore = clampv(
                    1.0 - std::abs(fill - 0.34) / 0.42,
                    0.0,
                    1.0);

                cv::Rect fullScaleBox(
                    static_cast<int>(std::floor(x / scale)),
                    static_cast<int>(std::floor(y / scale)),
                    static_cast<int>(std::ceil(w / scale)),
                    static_cast<int>(std::ceil(h / scale)));
                fullScaleBox &= cv::Rect(
                    0, 0, grayPatch.cols, grayPatch.rows);

                addCore(
                    fullScaleBox,
                    0.62 * centerScore + 0.28 * fillScore + 0.10 * (1.0 - fill));
            }
        }

        std::sort(
            cores.begin(),
            cores.end(),
            [](const Core& a, const Core& b) {
                return a.priority > b.priority;
            });
        if (static_cast<int>(cores.size()) > CORE_REFINE_MAX_CANDIDATES) {
            cores.resize(CORE_REFINE_MAX_CANDIDATES);
        }
        return cores;
    }
};


class NestedHCoreProbe
{
public:
    std::vector<HCoreRefiner::Core> refine(
        const cv::Mat& grayPatch,
        double proposalMajor) const
    {
        std::vector<HCoreRefiner::Core> cores;
        if (grayPatch.empty() || proposalMajor < STRICT_MIN_MAJOR_PX) {
            return cores;
        }

        const auto scales = corelock05_policy::nestedProbeScales();
        const double cx = 0.5 * grayPatch.cols;
        const double cy = 0.5 * grayPatch.rows;

        int rank = 0;
        for (double scale : scales) {
            if (rank >= NESTED_CORE_MAX_CHECKS) break;
            ++rank;

            const int w = std::max(6, static_cast<int>(std::lround(grayPatch.cols * scale)));
            const int h = std::max(6, static_cast<int>(std::lround(grayPatch.rows * scale)));
            cv::Rect box(
                static_cast<int>(std::lround(cx - 0.5 * w)),
                static_cast<int>(std::lround(cy - 0.5 * h)),
                w,
                h);
            box &= cv::Rect(0, 0, grayPatch.cols, grayPatch.rows);
            if (box.width < 6 || box.height < 6) continue;

            const double major = std::max(box.width, box.height);
            const double minor = std::min(box.width, box.height);
            if (major < FAR_MIN_MAJOR_PX || minor < FAR_MIN_MINOR_PX) continue;

            // Scale order encodes priority: 46% first, then smaller 36%, then
            // broader 58%. The H verifier still owns identity acceptance.
            const double priority = 1.0 - 0.10 * (rank - 1);
            cores.push_back({box, major, priority});
        }
        return cores;
    }
};


// ============================================================
// Rev11-native H-only local clearance evaluation
// ============================================================

class HClearanceVerifier
{
public:
    struct Result
    {
        double quietScore = 1.0; // retained name for downstream Rev11 weighting
        bool pass = true;
        bool hardChecked = false;
        bool bodyClean = true;
        bool haloClean = true;
        double bodyResidualRatio = 0.0;
        double bodyLargestComponentRatio = 0.0;
        double haloResidualRatio = 0.0;
        double haloLargestComponentRatio = 0.0;
    };

    Result evaluate(
        const cv::Mat& gray,
        const cv::Rect2f& targetBox,
        const cv::Point2f& targetCenter,
        double orientationDeg,
        bool darkPolarity,
        const cv::Mat& normalizedHMask,
        double internalClutterScore,
        double sourceMajorPx) const
    {
        Result result;

        pure_h_clearance::Metrics metrics;
        metrics.resolved = sourceMajorPx >= STRICT_MIN_MAJOR_PX;
        metrics.internalClutterScore = internalClutterScore;

        // FAR H keeps the trusted Rev11 temporal/structural behavior.  A 5%
        // halo is frequently sub-pixel here, so it is a soft score only.
        if (sourceMajorPx < STRICT_MIN_MAJOR_PX) {
            const auto policy = pure_h_clearance::evaluate(metrics);
            result.quietScore = policy.score;
            result.pass = true;
            result.hardChecked = false;
            result.bodyClean = true;
            result.haloClean = true;
            return result;
        }

        if (gray.empty() || targetBox.width < 4.0f || targetBox.height < 4.0f ||
            normalizedHMask.empty())
        {
            // Do not turn a missing context measurement into a new failure mode.
            // The H verifier has already established identity.
            metrics.resolved = false;
            const auto policy = pure_h_clearance::evaluate(metrics);
            result.quietScore = policy.score;
            result.pass = true;
            result.hardChecked = false;
            return result;
        }

        measureBodyResidual(normalizedHMask, metrics);
        measureHaloResidual(
            gray,
            targetBox,
            targetCenter,
            orientationDeg,
            darkPolarity,
            normalizedHMask,
            metrics);

        const auto policy = pure_h_clearance::evaluate(metrics);
        result.quietScore = policy.score;
        result.pass = policy.pass;
        result.hardChecked = policy.hardChecked;
        result.bodyClean = policy.bodyClean;
        result.haloClean = policy.haloClean;
        result.bodyResidualRatio = metrics.bodyResidualRatio;
        result.bodyLargestComponentRatio = metrics.bodyLargestComponentRatio;
        result.haloResidualRatio = metrics.haloResidualRatio;
        result.haloLargestComponentRatio = metrics.haloLargestComponentRatio;
        return result;
    }

private:
    struct ResidualStats
    {
        double ratio = 0.0;
        double largestRatio = 0.0;
    };

    static const std::vector<cv::Mat>& expectedHMasks()
    {
        static const std::vector<cv::Mat> masks = []() {
            std::vector<cv::Mat> result;
            for (double widthFraction : {0.10, 0.14, 0.18}) {
                cv::Mat expected = cv::Mat::zeros(CANON, CANON, CV_8UC1);
                const int bar = std::max(2, static_cast<int>(widthFraction * CANON));
                const int xLeft = static_cast<int>(0.24 * CANON);
                const int xRight = static_cast<int>(0.76 * CANON);
                const int yTop = static_cast<int>(0.08 * CANON);
                const int yBottom = static_cast<int>(0.92 * CANON);
                const int yMiddle = static_cast<int>(0.50 * CANON);

                cv::line(expected, cv::Point(xLeft, yTop),
                         cv::Point(xLeft, yBottom), cv::Scalar(255), bar, cv::LINE_8);
                cv::line(expected, cv::Point(xRight, yTop),
                         cv::Point(xRight, yBottom), cv::Scalar(255), bar, cv::LINE_8);
                cv::line(expected, cv::Point(xLeft, yMiddle),
                         cv::Point(xRight, yMiddle), cv::Scalar(255), bar, cv::LINE_8);
                result.push_back(std::move(expected));
            }
            return result;
        }();
        return masks;
    }

    static ResidualStats meaningfulResidual(
        const cv::Mat& mask,
        double denominator,
        int minMeaningfulArea)
    {
        ResidualStats out;
        if (mask.empty() || denominator <= 0.0) {
            return out;
        }

        cv::Mat labels, stats, centroids;
        const int count = cv::connectedComponentsWithStats(
            mask, labels, stats, centroids, 8, CV_32S);

        int total = 0;
        int largest = 0;
        for (int label = 1; label < count; ++label) {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area < minMeaningfulArea) {
                continue;
            }
            total += area;
            largest = std::max(largest, area);
        }

        out.ratio = total / denominator;
        out.largestRatio = largest / denominator;
        return out;
    }

    static void measureBodyResidual(
        const cv::Mat& normalizedHMask,
        pure_h_clearance::Metrics& metrics)
    {
        cv::Mat observed;
        if (normalizedHMask.size() != cv::Size(CANON, CANON)) {
            cv::resize(normalizedHMask, observed, cv::Size(CANON, CANON), 0, 0,
                       cv::INTER_NEAREST);
        }
        else {
            observed = normalizedHMask;
        }

        double bestCost = std::numeric_limits<double>::infinity();
        ResidualStats best;

        for (const cv::Mat& expected : expectedHMasks()) {
            cv::Mat tolerant;
            cv::dilate(
                expected,
                tolerant,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

            cv::Mat notExpected;
            cv::bitwise_not(tolerant, notExpected);
            cv::Mat residual;
            cv::bitwise_and(observed, notExpected, residual);

            const ResidualStats stats = meaningfulResidual(
                residual,
                static_cast<double>(CANON * CANON),
                3);
            const double cost = stats.ratio + 1.6 * stats.largestRatio;
            if (cost < bestCost) {
                bestCost = cost;
                best = stats;
            }
        }

        metrics.bodyResidualRatio = best.ratio;
        metrics.bodyLargestComponentRatio = best.largestRatio;
    }

    static cv::Size2d estimateOrientedSize(
        const cv::Rect2f& targetBox,
        double orientationDeg,
        const cv::Mat& normalizedHMask)
    {
        std::vector<cv::Point> points;
        cv::findNonZero(normalizedHMask, points);
        if (points.empty()) {
            return cv::Size2d(targetBox.width, targetBox.height);
        }

        const cv::Rect normBox = cv::boundingRect(points);
        const double ratio = clampv(
            static_cast<double>(normBox.width) / std::max(1, normBox.height),
            0.25,
            4.0);

        const double theta = orientationDeg * CV_PI / 180.0;
        const double c = std::abs(std::cos(theta));
        const double si = std::abs(std::sin(theta));
        const double aWidth = ratio * c + si;
        const double aHeight = ratio * si + c;
        const double denom = aWidth * aWidth + aHeight * aHeight;

        double h = denom > 1e-6
            ? (aWidth * targetBox.width + aHeight * targetBox.height) / denom
            : targetBox.height;
        h = std::max(4.0, h);
        double w = std::max(4.0, ratio * h);

        // Keep the reconstruction close to the measured source support.
        const double measuredMajor = std::max(
            static_cast<double>(targetBox.width),
            static_cast<double>(targetBox.height));
        const double reconstructedMajor = std::max(w, h);
        if (reconstructedMajor > 1e-6) {
            const double scale = clampv(measuredMajor / reconstructedMajor, 0.70, 1.45);
            w *= scale;
            h *= scale;
        }
        return cv::Size2d(w, h);
    }

    static void measureHaloResidual(
        const cv::Mat& gray,
        const cv::Rect2f& targetBox,
        const cv::Point2f& targetCenter,
        double orientationDeg,
        bool darkPolarity,
        const cv::Mat& normalizedHMask,
        pure_h_clearance::Metrics& metrics)
    {
        const cv::Size2d oriented = estimateOrientedSize(
            targetBox, orientationDeg, normalizedHMask);
        const double legHeight = std::max(4.0, oriented.height);
        const double haloPx = std::max(
            2.0,
            std::ceil(PURE_H_CLEARANCE_FRACTION * legHeight));
        const double pad = std::ceil(1.6 * haloPx + 4.0);

        const cv::Rect outer = clampRect(
            cv::Rect2f(
                static_cast<float>(targetBox.x - pad),
                static_cast<float>(targetBox.y - pad),
                static_cast<float>(targetBox.width + 2.0 * pad),
                static_cast<float>(targetBox.height + 2.0 * pad)),
            gray.size());

        if (outer.width < 8 || outer.height < 8) {
            return;
        }

        const cv::Mat originalLocal = gray(outer);
        const int maxDim = std::max(outer.width, outer.height);
        const double scale = maxDim > CLEARANCE_WORK_MAX_DIM
            ? static_cast<double>(CLEARANCE_WORK_MAX_DIM) / maxDim
            : 1.0;

        cv::Mat local;
        if (scale < 0.999) {
            cv::resize(originalLocal, local, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        else {
            local = originalLocal;
        }

        const cv::Point2f centerLocal(
            static_cast<float>((targetCenter.x - outer.x) * scale),
            static_cast<float>((targetCenter.y - outer.y) * scale));

        const cv::Mat rotation = cv::getRotationMatrix2D(
            centerLocal, orientationDeg, 1.0);
        cv::Mat rotated;
        cv::warpAffine(
            local,
            rotated,
            rotation,
            local.size(),
            cv::INTER_LINEAR,
            cv::BORDER_REPLICATE);

        cv::Mat smooth;
        cv::GaussianBlur(rotated, smooth, cv::Size(3, 3), 0.0);
        cv::Mat scratch;
        const double otsu = cv::threshold(
            smooth, scratch, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        cv::Mat foreground;
        cv::threshold(
            smooth,
            foreground,
            clampv(otsu, 1.0, 254.0),
            255,
            darkPolarity ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);

        const double w = oriented.width * scale;
        const double h = oriented.height * scale;
        const double halo = std::max(1.0, haloPx * scale);
        const double bleed = std::max(1.0, 0.012 * h);

        const cv::Rect inner = centeredRect(
            centerLocal,
            std::max(4.0, w + 2.0 * bleed),
            std::max(4.0, h + 2.0 * bleed),
            rotated.size());
        const cv::Rect outerHalo = centeredRect(
            centerLocal,
            std::max(6.0, w + 2.0 * (bleed + halo)),
            std::max(6.0, h + 2.0 * (bleed + halo)),
            rotated.size());

        if (inner.area() <= 0 || outerHalo.area() <= inner.area()) {
            return;
        }

        cv::Mat haloMask = cv::Mat::zeros(rotated.size(), CV_8UC1);
        haloMask(outerHalo).setTo(255);
        haloMask(inner).setTo(0);

        cv::Mat haloForeground;
        cv::bitwise_and(foreground, haloMask, haloForeground);

        const double bodyArea = std::max(1.0, w * h);
        const int minMeaningfulArea = std::max(
            2,
            static_cast<int>(std::lround(0.006 * bodyArea)));
        const double haloArea = std::max(
            1.0,
            static_cast<double>(cv::countNonZero(haloMask)));

        const ResidualStats haloStats = meaningfulResidual(
            haloForeground,
            haloArea,
            minMeaningfulArea);
        metrics.haloResidualRatio = haloStats.ratio;
        metrics.haloLargestComponentRatio = haloStats.largestRatio;
    }
};

// ============================================================
// Candidate generator and verifier
// ============================================================

class IsolatedHDetector
{
public:
    DetectionOutput detect(
        const cv::Mat& gray,
        const cv::Rect& requestedRoi,
        bool globalPass,
        bool trackingMode,
        double expectedMajorPx = 0.0,
        bool temporalPass = false) const
    {
        DetectionOutput output;

        if (gray.empty()) {
            return output;
        }

        const cv::Rect roi =
            requestedRoi & cv::Rect(0, 0, gray.cols, gray.rows);

        if (roi.width < 8 || roi.height < 8) {
            return output;
        }

        const cv::Mat source = gray(roi);

        const int proposalMaxWidth = globalPass
            ? GLOBAL_MAX_WIDTH
            : (trackingMode
                ? TRACK_PROPOSAL_MAX_WIDTH
                : LOCAL_PROPOSAL_MAX_WIDTH);

        const double scale = source.cols > proposalMaxWidth
            ? static_cast<double>(proposalMaxWidth) / source.cols
            : 1.0;

        cv::Mat work;
        if (scale < 0.999) {
            cv::resize(
                source,
                work,
                cv::Size(),
                scale,
                scale,
                cv::INTER_AREA);
        }
        else {
            work = source;
        }

        const double expectedMajorWork = expectedMajorPx > 0.0
            ? expectedMajorPx * scale
            : 0.0;

        std::vector<Proposal> proposals = generateProposals(
            work,
            expectedMajorWork,
            trackingMode,
            !globalPass);

        const int limit = globalPass
            ? MAX_PROPOSALS_GLOBAL
            : (trackingMode
                ? MAX_PROPOSALS_TRACK
                : MAX_PROPOSALS_LOCAL);

        if (static_cast<int>(proposals.size()) > limit) {
            proposals.resize(static_cast<size_t>(limit));
        }

        double bestDebugScore = -1.0;
        int coreRefineProposalCount = 0;

        for (const Proposal& proposal : proposals) {
            const double invScale = 1.0 / scale;

            cv::Rect2f fullBox(
                static_cast<float>(roi.x + proposal.box.x * invScale),
                static_cast<float>(roi.y + proposal.box.y * invScale),
                static_cast<float>(proposal.box.width * invScale),
                static_cast<float>(proposal.box.height * invScale));

            fullBox = cv::Rect2f(clampRect(fullBox, gray.size()));

            const double major = rectMajor(fullBox);
            const double minor = rectMinor(fullBox);

            if (major < FAR_MIN_MAJOR_PX ||
                minor < FAR_MIN_MINOR_PX ||
                major > MAX_TARGET_FRACTION *
                    std::max(gray.cols, gray.rows))
            {
                continue;
            }

            const cv::Rect patchRect = expandRect(
                fullBox,
                gray.size(),
                1.45,
                16.0);

            const cv::Mat patch = gray(patchRect);
            HShapeVerifier::Result shape =
                shapeVerifier_.verify(patch, major);

            cv::Rect selectedPatchRect = patchRect;
            cv::Rect2f selectedBox = fullBox;
            cv::Rect2f supportBox = fullBox;
            double selectedSourceMajor = major;
            bool usedCoreRefinement = false;
            bool usedNestedRefinement = false;
            bool componentRefineAttempted = false;

            if (shape.foregroundBox.area() > 0) {
                supportBox = cv::Rect2f(
                    static_cast<float>(patchRect.x + shape.foregroundBox.x),
                    static_cast<float>(patchRect.y + shape.foregroundBox.y),
                    static_cast<float>(shape.foregroundBox.width),
                    static_cast<float>(shape.foregroundBox.height));
                supportBox = cv::Rect2f(clampRect(supportBox, gray.size()));
            }

            HIntegrityVerifier::Result integrity;
            HIdentitySupportAnalyzer::Result supportAnalysis;
            double foregroundMajor = major;
            bool coreScaleMismatch = false;
            if (shape.foregroundBox.area() > 0) {
                foregroundMajor = std::max(
                    static_cast<double>(shape.foregroundBox.width),
                    static_cast<double>(shape.foregroundBox.height));
                coreScaleMismatch =
                    shape.valid &&
                    major >= STRICT_MIN_MAJOR_PX &&
                    foregroundMajor >= FAR_MIN_MAJOR_PX &&
                    foregroundMajor / major < CORE_REFINE_SCALE_MISMATCH_RATIO;
            }

            bool supportHeavy = false;
            if (shape.valid && major >= STRICT_MIN_MAJOR_PX) {
                supportAnalysis = identitySupportAnalyzer_.evaluate(shape.normalizedMask);
                supportHeavy = supportAnalysis.supportHeavy;
            }
            const bool canonicalizeIdentity =
                corelock04_policy::shouldCanonicalizeIdentity(
                    major >= STRICT_MIN_MAJOR_PX,
                    supportHeavy,
                    coreScaleMismatch);

            bool shapeReady = false;
            if (shape.valid) {
                integrity = integrityVerifier_.evaluate(
                    shape.normalizedMask, major);
                shapeReady = integrity.pass && !canonicalizeIdentity;
            }

            // Keep the real-field Rev11 fast path untouched. Refinement is a
            // bounded fallback only when a resolved outer proposal fails H
            // verification/integrity OR its verified mask still carries
            // substantial surrounding support that would bloat active identity.
            if (!shapeReady &&
                major >= STRICT_MIN_MAJOR_PX &&
                coreRefineProposalCount < CORE_REFINE_MAX_PROPOSALS_PER_PASS)
            {
                componentRefineAttempted = true;
                ++coreRefineProposalCount;
                const std::vector<HCoreRefiner::Core> cores =
                    coreRefiner_.refine(patch, shape, major);

                double bestCoreScore = -1.0;
                for (const HCoreRefiner::Core& core : cores) {
                    const cv::Rect localCorePatch = expandRect(
                        cv::Rect2f(
                            static_cast<float>(core.box.x),
                            static_cast<float>(core.box.y),
                            static_cast<float>(core.box.width),
                            static_cast<float>(core.box.height)),
                        patch.size(),
                        1.28,
                        14.0);
                    if (localCorePatch.width < 6 || localCorePatch.height < 6) {
                        continue;
                    }

                    const cv::Mat corePatch = patch(localCorePatch);
                    HShapeVerifier::Result coreShape =
                        shapeVerifier_.verify(corePatch, core.major);
                    if (!coreShape.valid) {
                        continue;
                    }

                    const HIntegrityVerifier::Result coreIntegrity =
                        integrityVerifier_.evaluate(
                            coreShape.normalizedMask, core.major);
                    if (!coreIntegrity.pass) {
                        continue;
                    }

                    const double coreScore = clampv(
                        0.88 * coreShape.shapeScore +
                        0.08 * coreIntegrity.score +
                        0.04 * clampv(core.priority, 0.0, 1.0),
                        0.0,
                        1.0);
                    if (coreScore <= bestCoreScore) {
                        continue;
                    }

                    bestCoreScore = coreScore;
                    shape = std::move(coreShape);
                    integrity = coreIntegrity;
                    selectedSourceMajor = core.major;
                    selectedPatchRect = cv::Rect(
                        patchRect.x + localCorePatch.x,
                        patchRect.y + localCorePatch.y,
                        localCorePatch.width,
                        localCorePatch.height);

                    if (shape.foregroundBox.area() > 0) {
                        selectedBox = cv::Rect2f(
                            static_cast<float>(selectedPatchRect.x + shape.foregroundBox.x),
                            static_cast<float>(selectedPatchRect.y + shape.foregroundBox.y),
                            static_cast<float>(shape.foregroundBox.width),
                            static_cast<float>(shape.foregroundBox.height));
                    }
                    else {
                        selectedBox = cv::Rect2f(
                            static_cast<float>(patchRect.x + core.box.x),
                            static_cast<float>(patchRect.y + core.box.y),
                            static_cast<float>(core.box.width),
                            static_cast<float>(core.box.height));
                    }
                    selectedBox = cv::Rect2f(clampRect(selectedBox, gray.size()));
                    usedCoreRefinement = true;
                    shapeReady = true;
                }
            }

            // CoreLock_05: if a resolved H is fused into one connected
            // support component, the component refiner has nothing useful to
            // separate. Probe a maximum of three centered inner crops, and let
            // the unchanged Rev11 H verifier decide whether one is genuine H.
            if (componentRefineAttempted &&
                corelock05_policy::shouldAttemptNestedProbe(
                    major >= STRICT_MIN_MAJOR_PX,
                    shapeReady,
                    usedCoreRefinement))
            {
                const std::vector<HCoreRefiner::Core> nestedCores =
                    nestedProbe_.refine(patch, major);
                double bestNestedScore = -1.0;
                int nestedChecks = 0;
                for (const HCoreRefiner::Core& core : nestedCores) {
                    if (nestedChecks >= NESTED_CORE_MAX_CHECKS) break;
                    ++nestedChecks;

                    const cv::Rect nestedRect = core.box &
                        cv::Rect(0, 0, patch.cols, patch.rows);
                    if (nestedRect.width < 6 || nestedRect.height < 6) continue;

                    const cv::Mat nestedPatch = patch(nestedRect);
                    HShapeVerifier::Result nestedShape =
                        shapeVerifier_.verify(nestedPatch, core.major);
                    if (!nestedShape.valid) continue;

                    double nestedMeasuredMajor = core.major;
                    if (nestedShape.foregroundBox.area() > 0) {
                        nestedMeasuredMajor = std::max(
                            static_cast<double>(nestedShape.foregroundBox.width),
                            static_cast<double>(nestedShape.foregroundBox.height));
                    }
                    if (nestedMeasuredMajor < FAR_MIN_MAJOR_PX) continue;

                    const HIntegrityVerifier::Result nestedIntegrity =
                        integrityVerifier_.evaluate(
                            nestedShape.normalizedMask,
                            nestedMeasuredMajor);
                    if (!nestedIntegrity.pass) continue;

                    const double nestedScore = clampv(
                        0.90 * nestedShape.shapeScore +
                        0.08 * nestedIntegrity.score +
                        0.02 * core.priority,
                        0.0, 1.0);
                    if (nestedScore <= bestNestedScore) continue;

                    bestNestedScore = nestedScore;
                    shape = std::move(nestedShape);
                    integrity = nestedIntegrity;
                    selectedSourceMajor = nestedMeasuredMajor;
                    selectedPatchRect = cv::Rect(
                        patchRect.x + nestedRect.x,
                        patchRect.y + nestedRect.y,
                        nestedRect.width,
                        nestedRect.height);
                    if (shape.foregroundBox.area() > 0) {
                        selectedBox = cv::Rect2f(
                            static_cast<float>(selectedPatchRect.x + shape.foregroundBox.x),
                            static_cast<float>(selectedPatchRect.y + shape.foregroundBox.y),
                            static_cast<float>(shape.foregroundBox.width),
                            static_cast<float>(shape.foregroundBox.height));
                    }
                    else {
                        selectedBox = cv::Rect2f(
                            static_cast<float>(patchRect.x + nestedRect.x),
                            static_cast<float>(patchRect.y + nestedRect.y),
                            static_cast<float>(nestedRect.width),
                            static_cast<float>(nestedRect.height));
                    }
                    selectedBox = cv::Rect2f(clampRect(selectedBox, gray.size()));
                    usedNestedRefinement = true;
                    shapeReady = true;
                }
            }

            if (!shapeReady) {
                continue;
            }

            const bool selectedFarMode =
                selectedSourceMajor < STRICT_MIN_MAJOR_PX;

            FarMicroHVerifier::Result farMicro;
            if (selectedSourceMajor < STRICT_MIN_MAJOR_PX) {
                farMicro = farMicroVerifier_.evaluate(shape.normalizedMask);
            }

            // Tiny single-frame shapes are intentionally held to a slightly
            // higher floor. A temporally aligned FAR candidate may use the
            // normal FAR threshold because evidence has already persisted
            // across several frames. This reduces tent/roof edge false locks.
            if (selectedFarMode && !temporalPass &&
                shape.shapeScore < FAR_SINGLE_FRAME_SHAPE_FLOOR)
            {
                continue;
            }

            cv::Point2f center(
                selectedPatchRect.x + shape.centerInPatch.x,
                selectedPatchRect.y + shape.centerInPatch.y);

            if (!selectedBox.contains(center)) {
                center = rectCenter(selectedBox);
            }

            cv::Rect2f clearanceBox = selectedBox;
            if (shape.foregroundBox.area() > 0) {
                clearanceBox = cv::Rect2f(
                    static_cast<float>(selectedPatchRect.x + shape.foregroundBox.x),
                    static_cast<float>(selectedPatchRect.y + shape.foregroundBox.y),
                    static_cast<float>(shape.foregroundBox.width),
                    static_cast<float>(shape.foregroundBox.height));
                clearanceBox = cv::Rect2f(clampRect(clearanceBox, gray.size()));
            }

            HClearanceVerifier::Result clearance = clearanceVerifier_.evaluate(
                gray,
                clearanceBox,
                center,
                shape.orientationDeg,
                shape.darkPolarity,
                shape.normalizedMask,
                shape.internalClutterScore,
                selectedSourceMajor);

            if (!clearance.pass) {
                continue;
            }

            const double continuationScore = trackingMode
                ? 1.0
                : externalContinuationScore(gray, selectedBox, center);
            const bool continuationPass =
                continuationScore >= EXTERNAL_CONTINUATION_PASS_MIN;

            const double resolutionScore = clampv(
                (selectedSourceMajor - FAR_MIN_MAJOR_PX) /
                std::max(1.0, STRICT_MIN_MAJOR_PX - FAR_MIN_MAJOR_PX),
                0.0,
                1.0);

            const double finalScore = clampv(
                0.75 * shape.shapeScore +
                0.11 * clearance.quietScore +
                0.05 * resolutionScore +
                0.04 * continuationScore,
                0.0,
                1.0);

            Candidate candidate;
            candidate.center = center;
            candidate.identityBox = selectedBox;
            candidate.supportBox = supportBox;
            candidate.box = candidate.identityBox;
            candidate.ellipseScore = clearance.quietScore;
            candidate.ringScore = 0.0;
            candidate.hScore = shape.shapeScore;
            candidate.finalScore = finalScore;
            candidate.acquisitionTopologyScore = shape.acquisitionTopologyScore;
            candidate.internalClutterScore = selectedFarMode
                ? shape.internalClutterScore
                : clearance.quietScore;
            candidate.externalContinuationScore = continuationScore;
            candidate.acquisitionTopologyPass = shape.acquisitionTopologyPass;
            candidate.acquisitionClutterPass =
                selectedFarMode ? shape.acquisitionClutterPass : clearance.pass;
            candidate.externalContinuationPass = continuationPass;
            candidate.darkPolarity = shape.darkPolarity;
            candidate.ringRadius = 0.0;
            candidate.hVerified = true;
            candidate.farMode = selectedFarMode;
            candidate.strictMode = !selectedFarMode;
            candidate.temporalSupported = temporalPass;
            if (candidate.farMode) {
                candidate.farSignatureKind = farMicro.signatureKind;
                candidate.farMicroAmbiguous = farMicro.ambiguous;
                candidate.farMicroPass = farMicro.pass;
                candidate.farMicroScore = farMicro.score;
                candidate.farMicroSideSupport = farMicro.metrics.sideSupport;
                candidate.farMicroBridgeSupport = farMicro.topology.bridgeStrength;
                candidate.farMicroGapEvidence = std::min(
                    farMicro.topology.gapTop, farMicro.topology.gapBottom);
                candidate.farMicroBlobCompactness = farMicro.topology.blobCompactness;
                candidate.farBridgeY = farMicro.topology.bridgeY;
                candidate.farLeftTopSupport = farMicro.topology.leftTop;
                candidate.farLeftBottomSupport = farMicro.topology.leftBottom;
                candidate.farRightTopSupport = farMicro.topology.rightTop;
                candidate.farRightBottomSupport = farMicro.topology.rightBottom;
                candidate.farLegTopSupport = 0.5 * (
                    farMicro.topology.leftTop + farMicro.topology.rightTop);
                candidate.farLegBottomSupport = 0.5 * (
                    farMicro.topology.leftBottom + farMicro.topology.rightBottom);
                candidate.farGapTop = farMicro.topology.gapTop;
                candidate.farGapBottom = farMicro.topology.gapBottom;
                candidate.farLeftRightBalance = farMicro.topology.leftRightBalance;
                candidate.farLargestComponentFraction =
                    farMicro.topology.largestComponentFraction;
                candidate.farTwoLegRowContinuity =
                    farMicro.topology.twoLegRowContinuity;
            }
            candidate.className = usedNestedRefinement
                ? "H-NESTED"
                : (usedCoreRefinement
                    ? "H-CORE"
                    : (supportHeavy ? "H-SUPPORT" : "H"));

            candidate.ellipseFull = cv::RotatedRect(
                center,
                cv::Size2f(
                    std::max(8.0f, selectedBox.width),
                    std::max(8.0f, selectedBox.height)),
                static_cast<float>(shape.orientationDeg));

            output.candidates.push_back(candidate);

            if (finalScore > bestDebugScore) {
                bestDebugScore = finalScore;
                output.bestPatch = gray(selectedPatchRect).clone();
                output.bestHMask = shape.normalizedMask.clone();
            }

            // During target-local tracking, one very strong identity is enough.
            if (trackingMode && finalScore >= 0.88) {
                break;
            }

            // A small/medium search may also stop on an excellent strict H. A
            // global pass continues so it can count multiple H candidates.
            if (!globalPass && !trackingMode &&
                shape.strictMode && finalScore >= 0.93)
            {
                break;
            }
        }

        std::sort(
            output.candidates.begin(),
            output.candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.finalScore > b.finalScore;
            });

        nonMaximumSuppress(output.candidates);

        if (static_cast<int>(output.candidates.size()) > MAX_ACCEPTED) {
            output.candidates.resize(MAX_ACCEPTED);
        }

        if (output.bestPatch.empty()) {
            output.bestPatch = makeTextImage("NO VALID H");
            output.bestHMask = makeTextImage("NO H MASK");
        }

        return output;
    }

    // Target-conditioned verification for an ALREADY ACQUIRED H. It uses a
    // compact dynamic patch around the predicted center and therefore does not
    // depend on the generic high-pass proposal scale on every frame.
    DetectionOutput trackKnownTarget(
        const cv::Mat& gray,
        const cv::Point2f& predictedCenter,
        const cv::Size2f& predictedSize,
        bool recovering) const
    {
        DetectionOutput output;

        if (gray.empty()) {
            return output;
        }

        const double predictedW = std::max(
            6.0, static_cast<double>(predictedSize.width));
        const double predictedH = std::max(
            6.0, static_cast<double>(predictedSize.height));
        const double predictedMajor = std::max(predictedW, predictedH);

        const double patchScale = recovering
            ? RECOVER_DIRECT_PATCH_SCALE
            : TRACK_DIRECT_PATCH_SCALE;

        const cv::Rect patchRect = centeredRect(
            predictedCenter,
            std::max(TRACK_DIRECT_MIN_PATCH, patchScale * predictedW),
            std::max(TRACK_DIRECT_MIN_PATCH, patchScale * predictedH),
            gray.size());

        if (patchRect.width < 8 || patchRect.height < 8) {
            return output;
        }

        const cv::Mat patch = gray(patchRect);
        HShapeVerifier::Result shape = shapeVerifier_.verify(
            patch,
            predictedMajor);

        cv::Rect selectedPatchRect = patchRect;
        cv::Rect2f selectedFallbackBox(
            static_cast<float>(predictedCenter.x - 0.5 * predictedW),
            static_cast<float>(predictedCenter.y - 0.5 * predictedH),
            static_cast<float>(predictedW),
            static_cast<float>(predictedH));
        selectedFallbackBox = cv::Rect2f(clampRect(selectedFallbackBox, gray.size()));
        double selectedSourceMajor = predictedMajor;
        bool usedCoreRefinement = false;
        bool usedNestedRefinement = false;
        bool componentRefineAttempted = false;

        cv::Rect2f supportBox = selectedFallbackBox;
        if (shape.valid && shape.foregroundBox.area() > 0) {
            supportBox = cv::Rect2f(
                static_cast<float>(patchRect.x + shape.foregroundBox.x),
                static_cast<float>(patchRect.y + shape.foregroundBox.y),
                static_cast<float>(shape.foregroundBox.width),
                static_cast<float>(shape.foregroundBox.height));
            supportBox = cv::Rect2f(clampRect(supportBox, gray.size()));
        }

        HIntegrityVerifier::Result integrity;
        HIdentitySupportAnalyzer::Result supportAnalysis;
        bool coreScaleMismatch = false;
        if (shape.valid && shape.foregroundBox.area() > 0) {
            const double foregroundMajor = std::max(
                static_cast<double>(shape.foregroundBox.width),
                static_cast<double>(shape.foregroundBox.height));
            coreScaleMismatch =
                predictedMajor >= STRICT_MIN_MAJOR_PX &&
                foregroundMajor >= FAR_MIN_MAJOR_PX &&
                foregroundMajor / predictedMajor < CORE_REFINE_SCALE_MISMATCH_RATIO;
        }

        bool supportHeavy = false;
        if (shape.valid && predictedMajor >= STRICT_MIN_MAJOR_PX) {
            supportAnalysis = identitySupportAnalyzer_.evaluate(shape.normalizedMask);
            supportHeavy = supportAnalysis.supportHeavy;
        }
        const bool canonicalizeIdentity =
            corelock04_policy::shouldCanonicalizeIdentity(
                predictedMajor >= STRICT_MIN_MAJOR_PX,
                supportHeavy,
                coreScaleMismatch);

        bool shapeReady = false;
        if (shape.valid && shape.shapeScore >= TRACK_DIRECT_SHAPE_MIN) {
            integrity = integrityVerifier_.evaluate(
                shape.normalizedMask, predictedMajor);
            shapeReady = integrity.pass && !canonicalizeIdentity;
        }

        // CoreLock_04: keep the successful real-field direct path untouched.
        // A resolved direct miss, scale mismatch, or support-heavy H+pad mask
        // gets one bounded H-core refinement inside the already-small local ROI.
        if (corelock03_policy::shouldAttemptDirectCoreRefine(
                predictedMajor >= STRICT_MIN_MAJOR_PX,
                shapeReady,
                coreScaleMismatch || supportHeavy))
        {
            componentRefineAttempted = true;
            const std::vector<HCoreRefiner::Core> cores =
                coreRefiner_.refine(patch, shape, predictedMajor);

            double bestCoreScore = -1.0;
            int coreChecks = 0;
            for (const HCoreRefiner::Core& core : cores) {
                if (coreChecks >= CORE_REFINE_MAX_CANDIDATES) break;
                ++coreChecks;

                const cv::Rect localCorePatch = expandRect(
                    cv::Rect2f(
                        static_cast<float>(core.box.x),
                        static_cast<float>(core.box.y),
                        static_cast<float>(core.box.width),
                        static_cast<float>(core.box.height)),
                    patch.size(),
                    1.28,
                    14.0);
                if (localCorePatch.width < 6 || localCorePatch.height < 6) {
                    continue;
                }

                const cv::Mat corePatch = patch(localCorePatch);
                HShapeVerifier::Result coreShape =
                    shapeVerifier_.verify(corePatch, core.major);
                if (!coreShape.valid ||
                    coreShape.shapeScore < TRACK_DIRECT_SHAPE_MIN)
                {
                    continue;
                }

                const HIntegrityVerifier::Result coreIntegrity =
                    integrityVerifier_.evaluate(
                        coreShape.normalizedMask, core.major);
                if (!coreIntegrity.pass) {
                    continue;
                }

                const double coreScore = clampv(
                    0.90 * coreShape.shapeScore +
                    0.07 * coreIntegrity.score +
                    0.03 * clampv(core.priority, 0.0, 1.0),
                    0.0,
                    1.0);
                if (coreScore <= bestCoreScore) {
                    continue;
                }

                bestCoreScore = coreScore;
                shape = std::move(coreShape);
                integrity = coreIntegrity;
                selectedSourceMajor = core.major;
                selectedPatchRect = cv::Rect(
                    patchRect.x + localCorePatch.x,
                    patchRect.y + localCorePatch.y,
                    localCorePatch.width,
                    localCorePatch.height);
                selectedFallbackBox = cv::Rect2f(
                    static_cast<float>(patchRect.x + core.box.x),
                    static_cast<float>(patchRect.y + core.box.y),
                    static_cast<float>(core.box.width),
                    static_cast<float>(core.box.height));
                selectedFallbackBox = cv::Rect2f(
                    clampRect(selectedFallbackBox, gray.size()));
                usedCoreRefinement = true;
                shapeReady = true;

                output.bestPatch = corePatch.clone();
                output.bestHMask = shape.normalizedMask.clone();
            }
        }

        if (componentRefineAttempted &&
            corelock05_policy::shouldAttemptNestedProbe(
                predictedMajor >= STRICT_MIN_MAJOR_PX,
                shapeReady,
                usedCoreRefinement))
        {
            const std::vector<HCoreRefiner::Core> nestedCores =
                nestedProbe_.refine(patch, predictedMajor);
            double bestNestedScore = -1.0;
            int nestedChecks = 0;
            for (const HCoreRefiner::Core& core : nestedCores) {
                if (nestedChecks >= NESTED_CORE_MAX_CHECKS) break;
                ++nestedChecks;

                const cv::Rect nestedRect = core.box &
                    cv::Rect(0, 0, patch.cols, patch.rows);
                if (nestedRect.width < 6 || nestedRect.height < 6) continue;

                const cv::Mat nestedPatch = patch(nestedRect);
                HShapeVerifier::Result nestedShape =
                    shapeVerifier_.verify(nestedPatch, core.major);
                if (!nestedShape.valid ||
                    nestedShape.shapeScore < TRACK_DIRECT_SHAPE_MIN)
                {
                    continue;
                }

                double nestedMeasuredMajor = core.major;
                if (nestedShape.foregroundBox.area() > 0) {
                    nestedMeasuredMajor = std::max(
                        static_cast<double>(nestedShape.foregroundBox.width),
                        static_cast<double>(nestedShape.foregroundBox.height));
                }
                if (nestedMeasuredMajor < FAR_MIN_MAJOR_PX) continue;

                const HIntegrityVerifier::Result nestedIntegrity =
                    integrityVerifier_.evaluate(
                        nestedShape.normalizedMask,
                        nestedMeasuredMajor);
                if (!nestedIntegrity.pass) continue;

                const double nestedScore = clampv(
                    0.91 * nestedShape.shapeScore +
                    0.07 * nestedIntegrity.score +
                    0.02 * core.priority,
                    0.0, 1.0);
                if (nestedScore <= bestNestedScore) continue;

                bestNestedScore = nestedScore;
                shape = std::move(nestedShape);
                integrity = nestedIntegrity;
                selectedSourceMajor = nestedMeasuredMajor;
                selectedPatchRect = cv::Rect(
                    patchRect.x + nestedRect.x,
                    patchRect.y + nestedRect.y,
                    nestedRect.width,
                    nestedRect.height);
                selectedFallbackBox = cv::Rect2f(
                    static_cast<float>(patchRect.x + nestedRect.x),
                    static_cast<float>(patchRect.y + nestedRect.y),
                    static_cast<float>(nestedRect.width),
                    static_cast<float>(nestedRect.height));
                selectedFallbackBox = cv::Rect2f(
                    clampRect(selectedFallbackBox, gray.size()));
                usedNestedRefinement = true;
                shapeReady = true;
                output.bestPatch = nestedPatch.clone();
                output.bestHMask = shape.normalizedMask.clone();
            }
        }

        if (!usedCoreRefinement && !usedNestedRefinement) {
            output.bestPatch = patch.clone();
            output.bestHMask = shape.normalizedMask.clone();
        }

        if (!shapeReady) {
            if (output.bestHMask.empty()) {
                output.bestHMask = makeTextImage("DIRECT H MISS");
            }
            return output;
        }

        cv::Point2f center(
            selectedPatchRect.x + shape.centerInPatch.x,
            selectedPatchRect.y + shape.centerInPatch.y);

        if (center.x < selectedPatchRect.x ||
            center.x >= selectedPatchRect.x + selectedPatchRect.width ||
            center.y < selectedPatchRect.y ||
            center.y >= selectedPatchRect.y + selectedPatchRect.height)
        {
            center = predictedCenter;
        }

        cv::Rect2f fullBox;
        if (shape.foregroundBox.area() > 0) {
            fullBox = cv::Rect2f(
                static_cast<float>(selectedPatchRect.x + shape.foregroundBox.x),
                static_cast<float>(selectedPatchRect.y + shape.foregroundBox.y),
                static_cast<float>(shape.foregroundBox.width),
                static_cast<float>(shape.foregroundBox.height));
            fullBox = cv::Rect2f(clampRect(fullBox, gray.size()));
        }
        else {
            fullBox = selectedFallbackBox;
        }

        const double measuredW = std::max(4.0, static_cast<double>(fullBox.width));
        const double measuredH = std::max(4.0, static_cast<double>(fullBox.height));
        const double wr = measuredW / predictedW;
        const double hr = measuredH / predictedH;
        if (wr < 0.42 || hr < 0.42 || wr > 3.6 || hr > 3.6) {
            return output;
        }

        const bool selectedFarMode =
            selectedSourceMajor < STRICT_MIN_MAJOR_PX;

        FarMicroHVerifier::Result farMicro;
        if (selectedFarMode) {
            farMicro = farMicroVerifier_.evaluate(shape.normalizedMask);
        }

        HClearanceVerifier::Result clearance = clearanceVerifier_.evaluate(
            gray,
            fullBox,
            center,
            shape.orientationDeg,
            shape.darkPolarity,
            shape.normalizedMask,
            shape.internalClutterScore,
            selectedSourceMajor);

        // Sticky Rev11 identity remains primary after acquisition. A strong,
        // integrity-verified H is not discarded because transient clutter enters
        // the small clearance halo for one frame.
        const bool contextOkay =
            clearance.pass ||
            shape.shapeScore >= TRACK_DIRECT_STRONG_SHAPE;

        if (!contextOkay) {
            return output;
        }

        const double finalScore = clampv(
            0.88 * shape.shapeScore +
            0.07 * clearance.quietScore,
            0.0,
            1.0);

        Candidate candidate;
        candidate.center = center;
        candidate.identityBox = fullBox;
        candidate.supportBox = supportBox;
        candidate.box = candidate.identityBox;
        candidate.ellipseScore = clearance.quietScore;
        candidate.ringScore = 0.0;
        candidate.hScore = shape.shapeScore;
        candidate.finalScore = finalScore;
        candidate.acquisitionTopologyScore = shape.acquisitionTopologyScore;
        candidate.internalClutterScore = selectedFarMode
            ? shape.internalClutterScore
            : clearance.quietScore;
        candidate.acquisitionTopologyPass = shape.acquisitionTopologyPass;
        candidate.acquisitionClutterPass =
            selectedFarMode ? shape.acquisitionClutterPass : clearance.pass;
        candidate.darkPolarity = shape.darkPolarity;
        candidate.ringRadius = 0.0;
        candidate.hVerified = true;
        candidate.farMode = selectedFarMode;
        candidate.strictMode = !selectedFarMode;
        if (candidate.farMode) {
            candidate.farSignatureKind = farMicro.signatureKind;
            candidate.farMicroAmbiguous = farMicro.ambiguous;
            candidate.farMicroPass = farMicro.pass;
            candidate.farMicroScore = farMicro.score;
            candidate.farMicroSideSupport = farMicro.metrics.sideSupport;
            candidate.farMicroBridgeSupport = farMicro.topology.bridgeStrength;
            candidate.farMicroGapEvidence = std::min(
                farMicro.topology.gapTop, farMicro.topology.gapBottom);
            candidate.farMicroBlobCompactness = farMicro.topology.blobCompactness;
            candidate.farBridgeY = farMicro.topology.bridgeY;
            candidate.farLeftTopSupport = farMicro.topology.leftTop;
            candidate.farLeftBottomSupport = farMicro.topology.leftBottom;
            candidate.farRightTopSupport = farMicro.topology.rightTop;
            candidate.farRightBottomSupport = farMicro.topology.rightBottom;
            candidate.farLegTopSupport = 0.5 * (
                farMicro.topology.leftTop + farMicro.topology.rightTop);
            candidate.farLegBottomSupport = 0.5 * (
                farMicro.topology.leftBottom + farMicro.topology.rightBottom);
            candidate.farGapTop = farMicro.topology.gapTop;
            candidate.farGapBottom = farMicro.topology.gapBottom;
            candidate.farLeftRightBalance = farMicro.topology.leftRightBalance;
            candidate.farLargestComponentFraction =
                farMicro.topology.largestComponentFraction;
            candidate.farTwoLegRowContinuity =
                farMicro.topology.twoLegRowContinuity;
        }
        candidate.className = usedNestedRefinement
            ? "H+NESTED"
            : (usedCoreRefinement
                ? "H+CORE"
                : (supportHeavy ? "H+SUPPORT" : "H"));

        candidate.ellipseFull = cv::RotatedRect(
            center,
            cv::Size2f(
                std::max(8.0f, fullBox.width),
                std::max(8.0f, fullBox.height)),
            static_cast<float>(shape.orientationDeg));

        output.candidates.push_back(candidate);
        return output;
    }

private:
    // Acquisition-only negative geometry test. A real isolated painted H has
    // stroke endpoints near the target extent. Runway stripes, road markings,
    // roof edges and fence lines often create a local H-like crossing while one
    // or more lines continue far outside the candidate. We detect only those
    // long center-crossing continuations; ordinary surrounding texture is not a
    // hard reject. Score 1.0 = isolated / good, 0.0 = strong continuation.
    static double externalContinuationScore(
        const cv::Mat& gray,
        const cv::Rect2f& targetBox,
        const cv::Point2f& targetCenter)
    {
        if (gray.empty()) return 0.0;

        const double major = std::max(8.0, rectMajor(targetBox));
        const double minor = std::max(4.0, rectMinor(targetBox));
        const cv::Rect outer = expandRect(targetBox, gray.size(), 2.60, 24.0);
        if (outer.width < 12 || outer.height < 12) return 1.0;

        cv::Mat patch = gray(outer);
        cv::Mat smooth, edges;
        cv::GaussianBlur(patch, smooth, cv::Size(3,3), 0.0);
        cv::Canny(smooth, edges, 45.0, 120.0, 3, true);

        std::vector<cv::Vec4i> lines;
        const int houghVotes = std::max(10, static_cast<int>(0.28 * major));
        const double minLen = std::max(12.0, 1.30 * major);
        const double maxGap = std::max(3.0, 0.22 * minor);
        cv::HoughLinesP(
            edges, lines, 1.0, CV_PI / 180.0,
            houghVotes, minLen, maxGap);

        const cv::Point2f c(
            targetCenter.x - static_cast<float>(outer.x),
            targetCenter.y - static_cast<float>(outer.y));

        auto pointSegmentDistance = [](
            const cv::Point2f& p,
            const cv::Point2f& a,
            const cv::Point2f& b)
        {
            const cv::Point2f ab = b - a;
            const double denom = std::max(1e-6, static_cast<double>(ab.dot(ab)));
            const double t = clampv(
                static_cast<double>((p - a).dot(ab)) / denom,
                0.0, 1.0);
            const cv::Point2f q = a + ab * static_cast<float>(t);
            return cv::norm(p - q);
        };

        double worstContinuation = 0.0;
        for (const cv::Vec4i& l : lines) {
            const cv::Point2f a(static_cast<float>(l[0]), static_cast<float>(l[1]));
            const cv::Point2f b(static_cast<float>(l[2]), static_cast<float>(l[3]));
            const double len = cv::norm(b - a);
            if (len < 1.30 * major) continue;

            // Only a long line that actually traverses the H neighbourhood is
            // suspicious. Distant field/runway lines are left alone.
            const double d = pointSegmentDistance(c, a, b);
            const double centerGate = std::max(0.60 * major, 2.5 * minor);
            if (d > centerGate) continue;

            const double lenScore = clampv(
                (len / major - 1.30) / 1.25, 0.0, 1.0);
            const double centerScore = clampv(
                1.0 - d / centerGate, 0.0, 1.0);
            worstContinuation = std::max(
                worstContinuation, lenScore * centerScore);
        }

        return clampv(1.0 - worstContinuation, 0.0, 1.0);
    }

    struct Proposal
    {
        cv::Rect2f box;
        double priority = 0.0;
    };

    static double integralRectMean(
        const cv::Mat& integral,
        const cv::Rect& rect)
    {
        if (rect.width <= 0 || rect.height <= 0) {
            return 0.0;
        }

        const int x1 = rect.x;
        const int y1 = rect.y;
        const int x2 = rect.x + rect.width;
        const int y2 = rect.y + rect.height;

        const double sum =
            integral.at<double>(y2, x2) -
            integral.at<double>(y1, x2) -
            integral.at<double>(y2, x1) +
            integral.at<double>(y1, x1);

        return sum /
            std::max(1.0, static_cast<double>(rect.area()));
    }

    static std::vector<Proposal> generateProposals(
        const cv::Mat& gray,
        double expectedMajorPx,
        bool trackingMode,
        bool allowBroadScale)
    {
        cv::Mat blur;
        cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0.0);

        const double adaptiveSigma = expectedMajorPx > 0.0
            ? clampv(
                TRACK_RESPONSE_SIGMA_GAIN * expectedMajorPx,
                3.0,
                TRACK_RESPONSE_SIGMA_MAX)
            : 3.0;

        cv::Mat background;
        cv::GaussianBlur(
            blur,
            background,
            cv::Size(0, 0),
            adaptiveSigma);

        cv::Mat response;
        cv::absdiff(blur, background, response);

        if (expectedMajorPx <= 0.0 && allowBroadScale) {
            cv::Mat broadBackground;
            cv::GaussianBlur(
                blur,
                broadBackground,
                cv::Size(0, 0),
                LOCAL_BROAD_RESPONSE_SIGMA);
            cv::Mat broadResponse;
            cv::absdiff(blur, broadBackground, broadResponse);
            cv::max(response, broadResponse, response);
        }

        cv::Scalar meanValue;
        cv::Scalar stdValue;
        cv::meanStdDev(response, meanValue, stdValue);

        const double threshold = std::max(
            4.0,
            meanValue[0] + (trackingMode ? 0.62 : 0.78) * stdValue[0]);

        cv::Mat binary;
        cv::threshold(
            response,
            binary,
            threshold,
            255,
            cv::THRESH_BINARY);

        int closeKernel = 3;
        if (trackingMode && expectedMajorPx > 16.0) {
            closeKernel = static_cast<int>(std::lround(0.045 * expectedMajorPx));
            closeKernel = clampv(closeKernel, 3, 5);
            if ((closeKernel % 2) == 0) ++closeKernel;
            closeKernel = std::min(closeKernel, 5);
        }

        cv::morphologyEx(
            binary,
            binary,
            cv::MORPH_CLOSE,
            cv::getStructuringElement(
                cv::MORPH_RECT,
                cv::Size(closeKernel, closeKernel)));

        // One light dilation is retained only for unknown-size discovery. During
        // tracking, the adaptive response already follows the H stroke width and
        // extra dilation would bias the measured growth.
        if (!trackingMode) {
            cv::dilate(
                binary,
                binary,
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
                cv::Point(-1, -1),
                1);
        }

        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;

        const int count = cv::connectedComponentsWithStats(
            binary,
            labels,
            stats,
            centroids,
            8,
            CV_32S);

        // Integral means avoid constructing a full-frame (labels == id) mask for
        // every component. That old O(component_count * image_area) operation was
        // the largest reason 4K video fell to 1-2 FPS.
        cv::Mat response64;
        response.convertTo(response64, CV_64F);
        cv::Mat integral;
        cv::integral(response64, integral, CV_64F);

        std::vector<Proposal> proposals;
        proposals.reserve(std::max(0, count - 1));

        const double imageArea =
            static_cast<double>(gray.cols) * gray.rows;
        const cv::Rect imageRect(0, 0, gray.cols, gray.rows);

        for (int label = 1; label < count; ++label) {
            const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(label, cv::CC_STAT_TOP);
            const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
            const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);

            if (area < 5 ||
                width < 3 || height < 3 ||
                static_cast<double>(width * height) > 0.20 * imageArea)
            {
                continue;
            }

            const double aspect =
                static_cast<double>(std::max(width, height)) /
                std::max(1.0, static_cast<double>(std::min(width, height)));

            if (aspect > 6.5) {
                continue;
            }

            const cv::Rect box(x, y, width, height);
            const cv::Point2f center(
                x + 0.5f * width,
                y + 0.5f * height);

            cv::Rect outer = centeredRect(
                center,
                2.6 * width,
                2.6 * height,
                gray.size());
            outer &= imageRect;

            const double insideMean = integralRectMean(integral, box) / 255.0;
            const double outerMeanAll = integralRectMean(integral, outer) / 255.0;

            const double outerArea = std::max(1.0, static_cast<double>(outer.area()));
            const double boxArea = std::max(1.0, static_cast<double>(box.area()));
            const double surroundingMean = outer.area() > box.area()
                ? clampv(
                    (outerMeanAll * outerArea - insideMean * boxArea) /
                    std::max(1.0, outerArea - boxArea),
                    0.0,
                    1.0)
                : 0.0;

            const double isolationScore = clampv(
                (insideMean - surroundingMean + 0.05) / 0.28,
                0.0,
                1.0);

            const double fill =
                static_cast<double>(area) /
                std::max(1.0, static_cast<double>(width * height));

            const double compactScore = clampv(
                1.0 - std::abs(fill - 0.34) / 0.40,
                0.0,
                1.0);

            const double sizeScore = clampv(
                std::sqrt(static_cast<double>(area)) / 22.0,
                0.0,
                1.0);

            Proposal proposal;
            proposal.box = cv::Rect2f(
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(width),
                static_cast<float>(height));

            const cv::Point2f imageCenter(
                0.5f * gray.cols,
                0.5f * gray.rows);
            const double centreNorm = std::max(
                1.0,
                0.5 * std::hypot(
                    static_cast<double>(gray.cols),
                    static_cast<double>(gray.rows)));
            const double centreScore = clampv(
                1.0 - cv::norm(center - imageCenter) / centreNorm,
                0.0,
                1.0);

            proposal.priority =
                0.32 * insideMean +
                0.28 * isolationScore +
                0.12 * compactScore +
                0.08 * sizeScore +
                0.20 * centreScore;

            proposals.push_back(proposal);
        }

        std::sort(
            proposals.begin(),
            proposals.end(),
            [](const Proposal& a, const Proposal& b) {
                return a.priority > b.priority;
            });

        return proposals;
    }

    static void nonMaximumSuppress(std::vector<Candidate>& candidates)
    {
        std::vector<Candidate> kept;
        kept.reserve(candidates.size());

        for (const Candidate& candidate : candidates) {
            bool duplicate = false;

            for (const Candidate& existing : kept) {
                const double size = std::max(
                    rectMajor(candidate.box),
                    rectMajor(existing.box));

                const double distance = cv::norm(
                    candidate.center - existing.center);

                if (distance < std::max(5.0, 0.38 * size)) {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate) {
                kept.push_back(candidate);
            }
        }

        candidates = std::move(kept);
    }

private:
    HShapeVerifier shapeVerifier_;
    HIntegrityVerifier integrityVerifier_;
    FarMicroHVerifier farMicroVerifier_;
    HIdentitySupportAnalyzer identitySupportAnalyzer_;
    HCoreRefiner coreRefiner_;
    NestedHCoreProbe nestedProbe_;
    HClearanceVerifier clearanceVerifier_;
};

// ============================================================
// Global frame-motion estimation
// ============================================================

struct FrameMotion
{
    bool valid = false;
    cv::Matx23f prevToCurrent{1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f};
    double dxPx = 0.0;
    double dyPx = 0.0;
    double rollDeltaDeg = 0.0;
    double scale = 1.0;
    double inlierRatio = 0.0;
};

static cv::Matx23f composeAffine(
    const cv::Matx23f& a,
    const cv::Matx23f& b)
{
    // Result maps a point through b first, then a.
    return cv::Matx23f(
        a(0,0)*b(0,0) + a(0,1)*b(1,0),
        a(0,0)*b(0,1) + a(0,1)*b(1,1),
        a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2),
        a(1,0)*b(0,0) + a(1,1)*b(1,0),
        a(1,0)*b(0,1) + a(1,1)*b(1,1),
        a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2));
}

static cv::Point2f transformPoint(
    const cv::Matx23f& m,
    const cv::Point2f& p)
{
    return cv::Point2f(
        m(0,0)*p.x + m(0,1)*p.y + m(0,2),
        m(1,0)*p.x + m(1,1)*p.y + m(1,2));
}

class GlobalMotionEstimator
{
public:
    // computeNow=false is the cheap in-between frame.  We advance the compact
    // reference image and feature locations using the last measured affine so
    // the next real LK update still compares adjacent frames.
    FrameMotion update(const cv::Mat& gray, bool computeNow = true)
    {
        FrameMotion out;
        if (gray.empty()) return out;

        const int maxWorkWidth = MOTION_WORK_MAX_WIDTH;
        const double scale = gray.cols > maxWorkWidth
            ? static_cast<double>(maxWorkWidth) / gray.cols
            : 1.0;

        cv::Mat current;
        if (scale < 0.999) {
            cv::resize(gray, current, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        else {
            current = gray;
        }

        if (previous_.empty() || previous_.size() != current.size()) {
            previous_ = current.clone();
            cachedPoints_.clear();
            lastMotion_ = {};
            return out;
        }

        if (!computeNow) {
            // Use the previous per-frame estimate as a low-cost prediction.
            out = lastMotion_;

            // Keep cached points approximately registered to the new reference
            // image so they can be reused at the next LK refresh.
            if (out.valid && !cachedPoints_.empty()) {
                cv::Matx23f small = out.prevToCurrent;
                small(0,2) = static_cast<float>(small(0,2) * scale);
                small(1,2) = static_cast<float>(small(1,2) * scale);
                for (cv::Point2f& p : cachedPoints_) {
                    p = transformPoint(small, p);
                }
            }

            previous_ = current.clone();
            return out;
        }

        std::vector<cv::Point2f> p0;
        if (cachedPoints_.size() >= static_cast<size_t>(MOTION_MIN_REUSE_FEATURES)) {
            p0 = cachedPoints_;
        }
        else {
            cv::goodFeaturesToTrack(
                previous_, p0, MOTION_MAX_FEATURES, 0.015, 12.0, cv::noArray(), 5);
        }

        if (p0.size() < 18) {
            previous_ = current.clone();
            cachedPoints_.clear();
            lastMotion_ = {};
            return out;
        }

        std::vector<cv::Point2f> p1;
        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(
            previous_, current, p0, p1, status, error,
            cv::Size(17,17), MOTION_LK_LEVELS,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 18, 0.015));

        std::vector<cv::Point2f> src, dst;
        src.reserve(p0.size());
        dst.reserve(p0.size());
        for (size_t i = 0; i < p0.size(); ++i) {
            if (!status[i] || error[i] > 35.0f) continue;
            const double d = cv::norm(p1[i] - p0[i]);
            if (d > 0.35 * std::hypot(current.cols, current.rows)) continue;
            src.push_back(p0[i]);
            dst.push_back(p1[i]);
        }

        previous_ = current.clone();
        cachedPoints_ = dst;

        if (src.size() < 14) {
            lastMotion_ = {};
            return out;
        }

        cv::Mat inliers;
        cv::Mat A = cv::estimateAffinePartial2D(
            src, dst, inliers, cv::RANSAC, 2.5, 700, 0.985, 10);

        if (A.empty() || A.rows != 2 || A.cols != 3) {
            lastMotion_ = {};
            return out;
        }

        const double a00 = A.at<double>(0,0);
        const double a01 = A.at<double>(0,1);
        const double a10 = A.at<double>(1,0);
        const double a11 = A.at<double>(1,1);
        const double txSmall = A.at<double>(0,2);
        const double tySmall = A.at<double>(1,2);

        int inlierCount = 0;
        if (!inliers.empty()) {
            const uchar* mask = inliers.ptr<uchar>(0);
            for (size_t i = 0; i < inliers.total(); ++i) {
                if (mask[i]) ++inlierCount;
            }
        }
        const double ratio = static_cast<double>(inlierCount) /
            std::max<size_t>(1, src.size());
        if (inlierCount < 10 || ratio < 0.42) {
            lastMotion_ = {};
            return out;
        }

        const double invScale = 1.0 / scale;
        out.valid = true;
        out.prevToCurrent = cv::Matx23f(
            static_cast<float>(a00), static_cast<float>(a01),
            static_cast<float>(txSmall * invScale),
            static_cast<float>(a10), static_cast<float>(a11),
            static_cast<float>(tySmall * invScale));
        out.dxPx = txSmall * invScale;
        out.dyPx = tySmall * invScale;
        out.rollDeltaDeg = std::atan2(a10, a00) * 180.0 / CV_PI;
        out.scale = std::sqrt(a00 * a00 + a10 * a10);
        out.inlierRatio = ratio;
        lastMotion_ = out;
        return out;
    }

private:
    cv::Mat previous_;
    std::vector<cv::Point2f> cachedPoints_;
    FrameMotion lastMotion_;
};

// ============================================================
// Search / tracking state machine
// ============================================================

} // namespace isolated_h_detail

class SearchTrack
{
public:
    enum class State
    {
        SEARCH,
        TRACK,
        RECOVER
    };

    struct Output
    {
        State state = State::SEARCH;

        bool hasBox = false;
        bool guidanceValid = false;
        bool observedThisFrame = false;
        bool predictedThisFrame = false;

        cv::Rect2f box;
        cv::Point2f center;
        cv::RotatedRect trackedEllipse;

        Candidate lastCandidate;

        int confirm = 0;
        int lost = 0;
        int identityMiss = 0;
        int candidateCount = 0;

        double activeConfidence = 0.0;
        double peakConfidence = 0.0;
        double confidenceDrop = 0.0;

        // Guidance quality is informational and does not change detector gates.
        // STRONG/WEAK are fresh observations, HOLD is the short verified-coordinate
        // hold, LOST means no guidance should be consumed.
        std::string quality = "LOST";
        std::string searchMode = "GLOBAL";
        std::vector<Candidate> candidates;
        std::vector<cv::Rect> activeTiles;

        cv::Mat debugPatch;
        cv::Mat debugH;
    };

    explicit SearchTrack(int /*reverifyEvery*/)
    {
    }

    State state() const { return state_; }
    bool hasPendingCandidate() const { return pendingActive_; }

    // V15.56 integration extension. An external system (for example GDINO/SAM2)
    // may nominate an ROI, but only a candidate already verified by the unchanged
    // PureH detector may enter this path. It does NOT bypass CoreLock acquisition:
    // the normal multi-observation pending/FOCUS confirmation still decides lock.
    bool offerExternalVerifiedCandidate(
        const cv::Mat& gray,
        const Candidate& candidate,
        bool centerPriorExempt = false)
    {
        if (gray.empty() || !candidate.hVerified || state_ != State::SEARCH)
            return false;

        std::vector<Candidate> candidates{candidate};
        lastCandidates_ = candidates;
        rankForAcquisition(candidates, gray.size());

        const bool wasPending = pendingActive_;
        const State oldState = state_;
        processAcquisitionCandidates(gray, candidates, centerPriorExempt);

        if (state_ == State::TRACK)
            searchMode_ = "EXT-ROI+CORE5-LOCK";
        else if (pendingActive_)
            searchMode_ = "EXT-ROI+CORE5-FOCUS";

        const bool accepted =
            state_ != oldState || (!wasPending && pendingActive_) ||
            (wasPending && pendingActive_);
        if (accepted)
            externalCandidateOfferedSinceUpdate_ = true;
        return accepted;
    }

    // Tiny/uncertain targets need frame-to-frame scene motion every processed
    // frame; resolved strong tracking may refresh motion every second frame.
    bool needsHighRateMotion() const
    {
        if (state_ != State::TRACK) return true;
        if (misses_ > 0) return true;
        return isolated_h_detail::rectMajor(activeBox_) <
            isolated_h_detail::FAR_TRACK_MAJOR_PX;
    }

    Output update(
        const cv::Mat& gray,
        const isolated_h_detail::FrameMotion& motion = {},
        bool detectionEnabled = true)
    {
        using namespace isolated_h_detail;

        ++frameId_;
        currentMotion_ = motion;
        observedThisFrame_ = false;
        predictedThisFrame_ = false;
        guidanceValid_ = false;
        lastCandidates_.clear();
        activeTiles_.clear();
        debugPatch_.release();
        debugH_.release();

        if (gray.empty()) {
            return makeOutput();
        }

        updateTemporalHistory(gray);

        // Optional FC pitch gating only suppresses initial SEARCH. If attitude
        // is unavailable, DetectionLogic passes detectionEnabled=true. Once an
        // H is locked TRACK/RECOVER continues even if pitch crosses the gate.
        if (!detectionEnabled && state_ == State::SEARCH) {
            searchMode_ = "ATTITUDE-GATED";
            return makeOutput();
        }

        // An externally nominated ROI was already independently PureH-verified
        // on this exact frame. Do not run FOCUS again on the same pixels and
        // accidentally count two confirmations from one image. The next update
        // performs normal CoreLock FOCUS / search on a genuinely new frame.
        if (externalCandidateOfferedSinceUpdate_) {
            externalCandidateOfferedSinceUpdate_ = false;
            searchMode_ += "+EXT-ONE-OBS";
            return makeOutput();
        }

        if (state_ == State::SEARCH && pendingActive_ &&
            focusFramesRemaining_ > 0)
        {
            runFocusedPendingSearch(gray);
            --focusFramesRemaining_;
        }
        else {
            const SearchTask task =
                isolated_h_detail::SEARCH_SCHEDULE[static_cast<size_t>(
                    scheduleIndex_ % isolated_h_detail::SCHEDULE_LENGTH)];

            scheduleIndex_ =
                (scheduleIndex_ + 1) % isolated_h_detail::SCHEDULE_LENGTH;

            if (state_ == State::SEARCH) {
                runSearchTask(gray, task);
            }
            else {
                runTrackTask(gray, task);
            }
        }

        return makeOutput();
    }

private:
    using SearchTask = isolated_h_detail::SearchTask;

    static const char* taskName(SearchTask task)
    {
        switch (task) {
            case SearchTask::SMALL:  return "SMALL";
            case SearchTask::MEDIUM: return "MEDIUM";
            case SearchTask::GLOBAL: return "GLOBAL";
        }
        return "?";
    }

    static double acquisitionRank(
        const Candidate& candidate,
        const cv::Size& frameSize)
    {
        const cv::Point2f frameCenter(
            0.5f * frameSize.width,
            0.5f * frameSize.height);
        const double norm = std::max(
            1.0,
            0.5 * std::hypot(
                static_cast<double>(frameSize.width),
                static_cast<double>(frameSize.height)));
        const double centreScore = isolated_h_detail::clampv(
            1.0 - cv::norm(candidate.center - frameCenter) / norm,
            0.0,
            1.0);

        // Shape remains dominant; centre position is a ranking prior, not an
        // identity gate.  An off-centre H can still win when it is clearly better.
        const double temporalBonus =
            (candidate.farMode && candidate.temporalSupported)
                ? isolated_h_detail::TEMPORAL_ACQUISITION_BONUS
                : 0.0;
        return candidate.finalScore +
            0.16 * centreScore +
            temporalBonus +
            0.04 * candidate.acquisitionTopologyScore +
            0.03 * candidate.internalClutterScore +
            0.03 * candidate.externalContinuationScore;
    }

    static void rankForAcquisition(
        std::vector<Candidate>& candidates,
        const cv::Size& frameSize)
    {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [&](const Candidate& a, const Candidate& b) {
                return acquisitionRank(a, frameSize) >
                       acquisitionRank(b, frameSize);
            });
    }

    Output makeOutput() const
    {
        Output output;
        output.state = state_;
        output.hasBox = active_ && (observedThisFrame_ || predictedThisFrame_);
        output.guidanceValid = guidanceValid_;
        output.observedThisFrame = observedThisFrame_;
        output.predictedThisFrame = predictedThisFrame_;
        output.box = activeBox_;
        output.center = activeCenter_;
        output.trackedEllipse = activeEllipse_;
        output.lastCandidate = lastCandidate_;
        output.confirm = pendingCount_;
        output.lost = misses_;
        output.identityMiss = misses_;
        output.candidateCount = lastGlobalCandidateCount_;
        output.activeConfidence = activeConfidence_;
        output.peakConfidence = activePeakConfidence_;
        output.confidenceDrop = activePeakConfidence_ > 1e-6
            ? isolated_h_detail::clampv(
                1.0 - activeConfidence_ / activePeakConfidence_,
                0.0,
                1.0)
            : 0.0;
        if (observedThisFrame_) {
            output.quality = activeConfidence_ >= 0.75 ? "STRONG" : "WEAK";
        }
        else if (predictedThisFrame_ && guidanceValid_) {
            output.quality = "PREDICT";
        }
        else {
            output.quality = "LOST";
        }
        output.searchMode = searchMode_;
        output.candidates = lastCandidates_;
        output.activeTiles = activeTiles_;
        output.debugPatch = debugPatch_.clone();
        output.debugH = debugH_.clone();
        return output;
    }

    cv::Rect nextGridRoi(
        const cv::Size& size,
        bool small)
    {
        // Drone guidance is camera-boresight driven. SMALL and MEDIUM use
        // explicit native-pixel dimensions configured above, not percentages.
        // centeredRect() automatically clips them when the source frame is
        // smaller than the requested window. GLOBAL is handled separately as
        // the full frame.
        const double width = small
            ? static_cast<double>(isolated_h_detail::SMALL_SEARCH_WIDTH_PX)
            : static_cast<double>(isolated_h_detail::MEDIUM_SEARCH_WIDTH_PX);
        const double height = small
            ? static_cast<double>(isolated_h_detail::SMALL_SEARCH_HEIGHT_PX)
            : static_cast<double>(isolated_h_detail::MEDIUM_SEARCH_HEIGHT_PX);

        return isolated_h_detail::centeredRect(
            cv::Point2f(0.5f * size.width, 0.5f * size.height),
            width,
            height,
            size);
    }

    cv::Size2f predictedTargetSize() const
    {
        using namespace isolated_h_detail;

        const double baseW = std::max(6.0, static_cast<double>(activeBox_.width));
        const double baseH = std::max(6.0, static_cast<double>(activeBox_.height));

        const double extraLookahead = state_ == State::RECOVER
            ? std::min(3.0, 0.75 * static_cast<double>(misses_))
            : 0.0;

        const double lookahead = SIZE_LOOKAHEAD_FRAMES + extraLookahead;

        const double predictedW = baseW * std::exp(clampv(
            widthLogRate_ * lookahead, -0.35, 0.65));
        const double predictedH = baseH * std::exp(clampv(
            heightLogRate_ * lookahead, -0.35, 0.65));

        return cv::Size2f(
            static_cast<float>(std::max(6.0, predictedW)),
            static_cast<float>(std::max(6.0, predictedH)));
    }

    cv::Rect targetCenteredRoi(
        const cv::Mat& gray,
        SearchTask task) const
    {
        using namespace isolated_h_detail;

        // Predict only the search-window centre. The committed target centre is
        // never advanced on a missed frame.
        const double centerLookahead = state_ == State::RECOVER
            ? 1.5 + 0.35 * std::min(misses_, 3)
            : 1.0;

        const cv::Point2f compensated = motionCompensatedPoint(activeCenter_);
        const cv::Point2f predictedCenter =
            compensated +
            activeVelocity_ * static_cast<float>(centerLookahead);

        const cv::Size2f predictedSize = predictedTargetSize();

        const double positiveGrowth = std::max(
            0.0, std::max(widthLogRate_, heightLogRate_));
        const double growthBonus = 1.0 + clampv(
            GROWTH_MARGIN_GAIN * positiveGrowth,
            0.0, MAX_GROWTH_MARGIN_BONUS);

        double marginScale = LOCAL_MARGIN_SCALE;
        if (task == SearchTask::MEDIUM) {
            marginScale = MEDIUM_MARGIN_SCALE;
        }

        const bool recovering = state_ == State::RECOVER;
        if (recovering) {
            marginScale = RECOVERY_MARGIN_SCALE +
                0.45 * std::min(misses_, 4);
        }

        double marginX = marginScale * predictedSize.width * growthBonus;
        double marginY = marginScale * predictedSize.height * growthBonus;

        // Expand only the axis that is moving. This costs fewer pixels than a
        // large square recovery window and keeps detector FPS high.
        marginX += MOTION_MARGIN_FRAMES * std::abs(activeVelocity_.x);
        marginY += MOTION_MARGIN_FRAMES * std::abs(activeVelocity_.y);

        double roiWidth = predictedSize.width + 2.0 * marginX;
        double roiHeight = predictedSize.height + 2.0 * marginY;

        const bool medium = task == SearchTask::MEDIUM || recovering;

        // Explicit pixel minima replace the old frame-percentage minima.
        // The dynamic target model may grow beyond these values when H becomes
        // larger, changes aspect ratio, or moves rapidly.
        const double minRoiWidth = medium
            ? static_cast<double>(MEDIUM_MIN_WIDTH_PX)
            : static_cast<double>(LOCAL_MIN_WIDTH_PX);
        const double minRoiHeight = medium
            ? static_cast<double>(MEDIUM_MIN_HEIGHT_PX)
            : static_cast<double>(LOCAL_MIN_HEIGHT_PX);

        roiWidth = std::max(roiWidth, std::min(minRoiWidth, static_cast<double>(gray.cols)));
        roiHeight = std::max(roiHeight, std::min(minRoiHeight, static_cast<double>(gray.rows)));

        const double maxWidthFrac = recovering
            ? RECOVERY_MAX_WIDTH_FRAC
            : LOCAL_MAX_WIDTH_FRAC;
        const double maxHeightFrac = recovering
            ? RECOVERY_MAX_HEIGHT_FRAC
            : LOCAL_MAX_HEIGHT_FRAC;

        roiWidth = std::min(roiWidth, maxWidthFrac * gray.cols);
        roiHeight = std::min(roiHeight, maxHeightFrac * gray.rows);

        return centeredRect(
            predictedCenter, roiWidth, roiHeight, gray.size());
    }

    void runSearchTask(const cv::Mat& gray, SearchTask task)
    {
        cv::Rect roi;
        bool globalPass = false;

        if (task == SearchTask::GLOBAL) {
            roi = cv::Rect(0, 0, gray.cols, gray.rows);
            globalPass = true;
        }
        else {
            roi = nextGridRoi(gray.size(), task == SearchTask::SMALL);
        }

        searchMode_ = taskName(task);
        activeTiles_.push_back(roi);

        // Exactly one primary proposal pass for this SEARCH frame.
        DetectionOutput detection = detector_.detect(
            gray, roi, globalPass, false);

        acceptDebug(detection);
        lastCandidates_ = detection.candidates;

        if (globalPass) {
            lastGlobalCandidateCount_ =
                static_cast<int>(detection.candidates.size());
        }

        // Keep the 5-frame far-H capability, but avoid a second complete detector
        // pass whenever the current frame already produced a plausible FAR H.
        if (!globalPass && !lastCandidates_.empty() &&
            lastCandidates_.front().farMode &&
            (frameId_ % isolated_h_detail::TEMPORAL_VERIFY_EVERY_N_FRAMES) == 0)
        {
            cv::Mat fusedRoi = buildTemporalRoi(gray, roi);
            if (!fusedRoi.empty()) {
                const Candidate seed = lastCandidates_.front();
                const cv::Point2f localCenter(
                    seed.center.x - static_cast<float>(roi.x),
                    seed.center.y - static_cast<float>(roi.y));
                const cv::Size2f seedSize(
                    std::max(6.0f, seed.box.width),
                    std::max(6.0f, seed.box.height));

                DetectionOutput temporal = detector_.trackKnownTarget(
                    fusedRoi, localCenter, seedSize, false);
                offsetDetectionToFull(
                    temporal,
                    cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y)));

                for (Candidate& c : temporal.candidates) {
                    c.temporalSupported = true;
                }

                if (!temporal.candidates.empty()) {
                    acceptDebug(temporal);
                    lastCandidates_.insert(
                        lastCandidates_.end(),
                        temporal.candidates.begin(),
                        temporal.candidates.end());
                    searchMode_ += "+TEMP_VERIFY";
                }
            }
        }
        // If no single frame can even propose the H, retain the expensive fused
        // acquisition pass at a controlled cadence. This is the case where the
        // five aligned frames are genuinely needed to assemble partial H evidence.
        else if (!globalPass && lastCandidates_.empty() &&
            (frameId_ % isolated_h_detail::TEMPORAL_ACQUIRE_EVERY_N_FRAMES) == 0)
        {
            cv::Mat fusedRoi = buildTemporalRoi(gray, roi);
            if (!fusedRoi.empty()) {
                const cv::Rect localRoi(0, 0, fusedRoi.cols, fusedRoi.rows);
                DetectionOutput temporal = detector_.detect(
                    fusedRoi, localRoi, false, false, 0.0, true);
                offsetDetectionToFull(
                    temporal,
                    cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y)));

                for (Candidate& c : temporal.candidates) {
                    c.temporalSupported = true;
                }

                if (!temporal.candidates.empty()) {
                    acceptDebug(temporal);
                    lastCandidates_.insert(
                        lastCandidates_.end(),
                        temporal.candidates.begin(),
                        temporal.candidates.end());
                    searchMode_ += "+TEMP_ACQUIRE";
                }
            }
        }

        rankForAcquisition(lastCandidates_, gray.size());
        processAcquisitionCandidates(gray, lastCandidates_);
    }

    void runFocusedPendingSearch(const cv::Mat& gray)
    {
        const double size = std::max(
            12.0,
            isolated_h_detail::rectMajor(pendingCandidate_.box));

        const cv::Rect roi = isolated_h_detail::centeredRect(
            pendingCandidate_.center,
            std::max(7.0 * size, 0.22 * gray.cols),
            std::max(7.0 * size, 0.22 * gray.rows),
            gray.size());

        searchMode_ = "FOCUS";
        activeTiles_.push_back(roi);

        // Revision-6: pending identity is already known. Verify that exact
        // target first instead of re-running generic proposal generation and
        // accidentally switching to a nearby texture. Generic detection is only
        // a fallback if direct verification fails.
        const cv::Size2f pendingSize(
            std::max(6.0f, pendingCandidate_.box.width),
            std::max(6.0f, pendingCandidate_.box.height));
        DetectionOutput focused = detector_.trackKnownTarget(
            gray, pendingCandidate_.center, pendingSize, false);

        if (focused.candidates.empty()) {
            focused = detector_.detect(gray, roi, false, false);
        }

        acceptDebug(focused);
        lastCandidates_ = focused.candidates;

        // FAR pending confirmation keeps the full 5-frame history, but uses a
        // cheap target-conditioned temporal verifier instead of another generic
        // proposal pass. This preserves long-range evidence without restoring the
        // old FOCUS+CENTER double-detector cost.
        if (!lastCandidates_.empty() &&
            lastCandidates_.front().farMode &&
            !lastCandidates_.front().temporalSupported)
        {
            cv::Mat fusedRoi = buildTemporalRoi(gray, roi);
            if (!fusedRoi.empty()) {
                const Candidate seed = lastCandidates_.front();
                const cv::Point2f localCenter(
                    seed.center.x - static_cast<float>(roi.x),
                    seed.center.y - static_cast<float>(roi.y));
                const cv::Size2f seedSize(
                    std::max(6.0f, seed.box.width),
                    std::max(6.0f, seed.box.height));
                DetectionOutput temporal = detector_.trackKnownTarget(
                    fusedRoi, localCenter, seedSize, false);
                offsetDetectionToFull(
                    temporal,
                    cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y)));
                for (Candidate& c : temporal.candidates) {
                    c.temporalSupported = true;
                }
                lastCandidates_.insert(
                    lastCandidates_.end(),
                    temporal.candidates.begin(),
                    temporal.candidates.end());
                if (!temporal.candidates.empty()) searchMode_ += "+TEMP_VERIFY";
            }
        }

        rankForAcquisition(lastCandidates_, gray.size());
        processAcquisitionCandidates(gray, lastCandidates_);

        if (lastCandidates_.empty()) {
            ++pendingFocusMisses_;
            if (pendingFocusMisses_ > isolated_h_detail::PENDING_FOCUS_MISS_TOLERANCE) {
                focusFramesRemaining_ = 0;
            }
        }
        else {
            pendingFocusMisses_ = 0;
        }
    }

    void processAcquisitionCandidates(
        const cv::Mat& gray,
        const std::vector<Candidate>& candidates,
        bool centerPriorExempt = false)
    {
        ++pendingAge_;

        if (pendingAge_ > isolated_h_detail::PENDING_MAX_AGE) {
            clearPending();
        }

        if (candidates.empty()) {
            return;
        }

        const Candidate best = candidates.front();
        if (best.className.find("NESTED") != std::string::npos)
            searchMode_ += "+CORE=NESTED";
        if (best.farMode) {
            if (best.farSignatureKind == corelock05_policy::FarSignatureKind::Strong)
                searchMode_ += "+FAR_SIG=STRONG";
            else if (best.farSignatureKind == corelock05_policy::FarSignatureKind::Ambiguous)
                searchMode_ += "+FAR_SIG=AMBIG";
            else
                searchMode_ += "+FAR_SIG=REJECT";
        }

        const cv::Point2f frameCenter(
            0.5f * gray.cols,
            0.5f * gray.rows);
        const double centreNorm = std::max(
            1.0,
            0.5 * std::hypot(
                static_cast<double>(gray.cols),
                static_cast<double>(gray.rows)));
        const double centreScore = isolated_h_detail::clampv(
            1.0 - cv::norm(best.center - frameCenter) / centreNorm,
            0.0,
            1.0);

        // Revision-6: high shape score alone may not override the boresight prior.
        // Off-centre acquisition still needs explicit H topology, a sufficiently
        // clean raw component field, and strong H evidence. No ring/circle shortcut.
        const bool structuralPass =
            best.acquisitionTopologyPass &&
            best.acquisitionClutterPass &&
            best.externalContinuationPass;
        const bool farHOnlyOverrideAllowed =
            !best.farMode ||
            centreScore >= isolated_h_detail::FAR_H_ONLY_OVERRIDE_MIN_CENTER_SCORE;

        const bool offCentreOverride =
            structuralPass && farHOnlyOverrideAllowed &&
            best.hScore >= isolated_h_detail::OFFCENTER_H_OVERRIDE;

        if (!pendingActive_ &&
            !centerPriorExempt &&
            centreScore < isolated_h_detail::INITIAL_CENTER_SCORE_MIN &&
            !offCentreOverride)
        {
            return;
        }

        // Never commit flight guidance from a single image.  Two strict
        // observations are only about 33-67 ms at 60/30 FPS, but remove the
        // false one-frame locks visible in the real-drone recording.
        const bool immediateStrict = false;
        (void)immediateStrict;

        if (pendingActive_ && sameIdentity(best, pendingCandidate_)) {
            ++pendingCount_;

            if (best.farMode) {
                const bool currentUsable =
                    best.farSignatureKind != corelock05_policy::FarSignatureKind::Reject;
                const bool previousUsable =
                    pendingCandidate_.farMode &&
                    pendingCandidate_.farSignatureKind !=
                        corelock05_policy::FarSignatureKind::Reject;
                const bool compatible = currentUsable && previousUsable &&
                    corelock05_policy::farSignaturesCompatible(
                        candidateFarTopology(pendingCandidate_),
                        pendingCandidate_.darkPolarity,
                        pendingCandidate_.ellipseFull.angle,
                        candidateFarTopology(best),
                        best.darkPolarity,
                        best.ellipseFull.angle);

                if (best.farSignatureKind ==
                        corelock05_policy::FarSignatureKind::Strong)
                {
                    pendingFarSignatureCompatCount_ = std::max(
                        pendingFarSignatureCompatCount_, 2);
                }
                else if (compatible) {
                    ++pendingFarSignatureCompatCount_;
                }
                else {
                    pendingFarSignatureCompatCount_ = currentUsable ? 1 : 0;
                }
            }
            else {
                pendingFarSignatureCompatCount_ = 2;
            }

            pendingCandidate_ = best;
            pendingTemporalSeen_ = pendingTemporalSeen_ || best.temporalSupported;
            if (best.acquisitionTopologyPass) ++pendingTopologyStrongCount_;
            if (best.temporalSupported && best.acquisitionTopologyPass)
                pendingTemporalTopologySeen_ = true;
            if (best.acquisitionClutterPass) ++pendingClutterCleanCount_;
            if (best.externalContinuationPass) ++pendingContinuationCleanCount_;
        }
        else {
            pendingCandidate_ = best;
            pendingActive_ = true;
            pendingCount_ = 1;
            pendingTemporalSeen_ = best.temporalSupported;
            pendingTemporalTopologySeen_ =
                best.temporalSupported && best.acquisitionTopologyPass;
            pendingTopologyStrongCount_ = best.acquisitionTopologyPass ? 1 : 0;
            pendingClutterCleanCount_ = best.acquisitionClutterPass ? 1 : 0;
            pendingContinuationCleanCount_ = best.externalContinuationPass ? 1 : 0;
            if (!best.farMode ||
                best.farSignatureKind == corelock05_policy::FarSignatureKind::Strong)
            {
                pendingFarSignatureCompatCount_ = 2;
            }
            else if (best.farSignatureKind ==
                     corelock05_policy::FarSignatureKind::Ambiguous)
            {
                pendingFarSignatureCompatCount_ = 1;
            }
            else {
                pendingFarSignatureCompatCount_ = 0;
            }
        }

        pendingAge_ = 0;
        focusFramesRemaining_ = isolated_h_detail::FOCUS_CONFIRM_FRAMES;

        int required = best.strictMode
            ? isolated_h_detail::STRICT_CONFIRM_FRAMES
            : isolated_h_detail::FAR_CONFIRM_FRAMES;

        const bool farNeedsTemporal =
            best.farMode &&
            !pendingTemporalSeen_;

        const int topologyNeeded = best.farMode
            ? isolated_h_detail::FAR_TOPOLOGY_CONFIRM_MIN
            : isolated_h_detail::STRICT_TOPOLOGY_CONFIRM_MIN;
        const int clutterNeeded = best.farMode
            ? isolated_h_detail::FAR_CLUTTER_CONFIRM_MIN
            : isolated_h_detail::STRICT_CLUTTER_CONFIRM_MIN;
        const int continuationNeeded = best.farMode
            ? isolated_h_detail::FAR_CONTINUATION_CONFIRM_MIN
            : isolated_h_detail::STRICT_CONTINUATION_CONFIRM_MIN;

        // FAR topology is allowed to come from one aligned temporal observation:
        // that observation already represents evidence accumulated across the
        // 5-frame buffer, which is exactly what tiny H acquisition needs.
        const bool topologyReady = best.farMode
            ? (pendingTemporalTopologySeen_ ||
               pendingTopologyStrongCount_ >= topologyNeeded)
            : (pendingTopologyStrongCount_ >= topologyNeeded);

        const bool acquisitionStructureReady =
            topologyReady &&
            pendingClutterCleanCount_ >= clutterNeeded &&
            pendingContinuationCleanCount_ >= continuationNeeded;

        const bool farMicroReady =
            !best.farMode ||
            corelock05_policy::farSignatureConsensusReady(
                best.farSignatureKind,
                pendingFarSignatureCompatCount_);

        if (pendingCount_ >= required &&
            !farNeedsTemporal &&
            acquisitionStructureReady &&
            farMicroReady)
        {
            lockCandidate(gray, best);
        }
        else if (pendingCount_ >= required + 2 &&
                 (!acquisitionStructureReady || !farMicroReady)) {
            // Do not let a persistent bush/tree/road texture monopolize FOCUS.
            // CoreLock_05 also releases a FAR candidate whose structural
            // signature never becomes compatible enough to earn identity.
            clearPending();
        }
    }

    void runTrackTask(const cv::Mat& gray, SearchTask task)
    {
        const SearchTask localTask = task == SearchTask::GLOBAL
            ? SearchTask::SMALL
            : task;

        const cv::Rect localRoi = targetCenteredRoi(gray, localTask);
        searchMode_ = "STICKY";
        activeTiles_.push_back(localRoi);

        const cv::Point2f framePredicted = motionCompensatedPoint(activeCenter_);
        const cv::Point2f predicted = framePredicted + activeVelocity_;
        const cv::Size2f predictedSize = predictedTargetSize();
        const double expectedMajor = std::max(
            static_cast<double>(predictedSize.width),
            static_cast<double>(predictedSize.height));

        // Normal Revision-6 authority: verify the known H first on every frame.
        DetectionOutput local = detector_.trackKnownTarget(
            gray, predicted, predictedSize, state_ == State::RECOVER);

        std::optional<Candidate> activeMatch =
            chooseActiveMatch(local.candidates, predicted);

        acceptDebug(local);
        lastCandidates_ = local.candidates;

        const bool farTrack =
            expectedMajor < isolated_h_detail::FAR_TRACK_MAJOR_PX;

        // Keep the existing FAR target-conditioned temporal identity check.
        if (farTrack &&
            (frameId_ % isolated_h_detail::TEMPORAL_VERIFY_EVERY_N_FRAMES) == 0)
        {
            cv::Mat fusedRoi = buildTemporalRoi(gray, localRoi);
            if (!fusedRoi.empty()) {
                const cv::Point2f localPredicted(
                    predicted.x - static_cast<float>(localRoi.x),
                    predicted.y - static_cast<float>(localRoi.y));
                DetectionOutput temporal = detector_.trackKnownTarget(
                    fusedRoi, localPredicted, predictedSize,
                    state_ == State::RECOVER);
                offsetDetectionToFull(
                    temporal,
                    cv::Point2f(
                        static_cast<float>(localRoi.x),
                        static_cast<float>(localRoi.y)));
                for (Candidate& c : temporal.candidates)
                    c.temporalSupported = true;

                std::optional<Candidate> temporalMatch =
                    chooseActiveMatch(temporal.candidates, predicted);
                if (temporalMatch &&
                    (!activeMatch ||
                     temporalMatch->finalScore > activeMatch->finalScore + 0.02))
                {
                    activeMatch = temporalMatch;
                    acceptDebug(temporal);
                }
                local.candidates.insert(
                    local.candidates.end(),
                    temporal.candidates.begin(),
                    temporal.candidates.end());
                if (!temporal.candidates.empty())
                    searchMode_ = "FAR-TRACK+TEMP";
            }
        }

        if (activeProvisional_ && activeMatch &&
            !provisionalCandidateCompatible(*activeMatch))
        {
            activeMatch.reset();
            searchMode_ += "+PROVISIONAL-MISMATCH";
        }

        if (activeMatch &&
            activeMatch->className.find("NESTED") != std::string::npos)
        {
            searchMode_ += "+CORE=NESTED";
        }

        const bool strongDirect = activeMatch &&
            activeMatch->hScore >= isolated_h_detail::TRACK_DIRECT_STRONG_SHAPE &&
            activeMatch->finalScore >= 0.75;

        if (!farTrack && strongDirect) ++strongTrackStreak_;
        else strongTrackStreak_ = 0;

        if (!farTrack && strongDirect &&
            strongTrackStreak_ >= isolated_h_detail::STRONG_TRACK_CONFIRM_FRAMES)
        {
            searchMode_ = "STICKY-FAST";
            directHMissStreak_ = 0;
            clearEscape();
            updateActive(gray, *activeMatch);
            std::vector<Candidate> onlyActive{*activeMatch};
            processChallenger(onlyActive, activeMatch);
            return;
        }

        std::vector<Candidate> decisionCandidates = local.candidates;

        if (activeMatch) {
            directHMissStreak_ = 0;
            clearEscape();
            updateActive(gray, *activeMatch);
            processChallenger(decisionCandidates, activeMatch);
            return;
        }

        ++directHMissStreak_;

        // CoreLock_05 provisional FAR identity is intentionally short-lived.
        // It may coast for one frame with measured frame motion, but it cannot
        // seed StrictTracker or enter the normal staged recovery machinery.
        if (activeProvisional_) {
            const int priorMisses = provisionalMissStreak_;
            ++provisionalMissStreak_;
            recoveryTracker_.reset();
            recoveryTrackerSeedAge_ = 0;

            if (corelock05_policy::provisionalPredictionAllowed(priorMisses) &&
                applyFrameMotionContinuation(gray, framePredicted))
            {
                searchMode_ = "PROVISIONAL-PREDICT";
                return;
            }

            reset();
            searchMode_ = "PROVISIONAL-DROP";
            return;
        }

        // ================================================================
        // Revision-10 FLUENT miss path
        // ================================================================
        // A verified lock is first continued with measured frame motion. This
        // keeps guidance fluent through blur/jitter without letting prediction
        // masquerade as a fresh H observation. The direct H verifier above still
        // runs EVERY frame. While prediction is healthy, only scheduled MEDIUM
        // or GLOBAL generic work is allowed; SMALL frames stay cheap.
        const bool frameMotionMoved = applyFrameMotionContinuation(gray, framePredicted);

        if (frameMotionMoved) {
            std::optional<Candidate> scheduledMatch =
                runPredictedScheduledSearch(
                    gray, task, expectedMajor, farTrack, decisionCandidates);

            if (scheduledMatch) {
                if (escapeRelocationReady_) {
                    relocateActiveVerified(gray, *scheduledMatch);
                }
                else {
                    updateActive(gray, *scheduledMatch);
                }
                processChallenger(decisionCandidates, scheduledMatch);
                return;
            }

            if (task != SearchTask::SMALL || !decisionCandidates.empty()) {
                processChallenger(decisionCandidates, std::nullopt);
            }
            return;
        }

        // If scene motion is unavailable/unreliable, retain the Revision-9
        // StrictTracker + staged recovery fallback. Tracker output alone never
        // becomes TRACK.
        const bool trackerMoved = applyRecoveryTracker(gray);

        if (!trackerMoved && active_) {
            handleLocalMiss(gray, predicted);
        }

        if (!active_) {
            processChallenger(decisionCandidates, std::nullopt);
            return;
        }

        if (trackerMoved) {
            const cv::Size2f trackerSize = predictedTargetSize();
            DetectionOutput trackerVerify = detector_.trackKnownTarget(
                gray, activeCenter_, trackerSize, true);
            std::optional<Candidate> trackerMatch =
                chooseActiveMatch(trackerVerify.candidates, activeCenter_);

            if (!trackerVerify.bestPatch.empty()) {
                acceptDebug(trackerVerify);
            }
            decisionCandidates.insert(
                decisionCandidates.end(),
                trackerVerify.candidates.begin(),
                trackerVerify.candidates.end());
            lastCandidates_ = trackerVerify.candidates;

            if (trackerMatch) {
                searchMode_ = "RECOVER+USER-TRACKER+H";
                updateActive(gray, *trackerMatch);
                processChallenger(decisionCandidates, trackerMatch);
                return;
            }
        }

        std::optional<Candidate> stagedMatch = runOneRecoveryStage(
            gray, expectedMajor, farTrack, trackerMoved, decisionCandidates);

        if (stagedMatch) {
            if (escapeRelocationReady_) {
                relocateActiveVerified(gray, *stagedMatch);
            }
            else {
                updateActive(gray, *stagedMatch);
            }
        }

        processChallenger(decisionCandidates, stagedMatch);
    }

    bool applyFrameMotionContinuation(
        const cv::Mat& gray,
        const cv::Point2f& framePredicted)
    {
        if (!active_ ||
            gray.empty() ||
            !currentMotion_.valid ||
            currentMotion_.inlierRatio < isolated_h_detail::FRAME_MOTION_MIN_INLIER ||
            misses_ >= isolated_h_detail::FRAME_MOTION_PREDICT_MAX_MISSES ||
            framePredicted.x < 0.0f || framePredicted.x >= gray.cols ||
            framePredicted.y < 0.0f || framePredicted.y >= gray.rows)
        {
            return false;
        }

        ++misses_;
        ++recoveryStageFrame_;
        ++frameMotionConsecutive_;

        activeCenter_ = framePredicted;
        activeVelocity_ *= 0.35f;

        const double frameScale = isolated_h_detail::clampv(
            currentMotion_.scale,
            isolated_h_detail::FRAME_MOTION_SCALE_MIN,
            isolated_h_detail::FRAME_MOTION_SCALE_MAX);

        activeBox_.width = static_cast<float>(isolated_h_detail::clampv(
            static_cast<double>(activeBox_.width) * frameScale,
            4.0,
            isolated_h_detail::MAX_TARGET_FRACTION * gray.cols));
        activeBox_.height = static_cast<float>(isolated_h_detail::clampv(
            static_cast<double>(activeBox_.height) * frameScale,
            4.0,
            isolated_h_detail::MAX_TARGET_FRACTION * gray.rows));
        activeBox_.x = activeCenter_.x - 0.5f * activeBox_.width;
        activeBox_.y = activeCenter_.y - 0.5f * activeBox_.height;

        activeEllipse_.center = activeCenter_;
        activeEllipse_.size.width = static_cast<float>(
            std::max(4.0, static_cast<double>(activeEllipse_.size.width) * frameScale));
        activeEllipse_.size.height = static_cast<float>(
            std::max(4.0, static_cast<double>(activeEllipse_.size.height) * frameScale));
        activeEllipse_.angle += static_cast<float>(currentMotion_.rollDeltaDeg);

        // Avoid double-counting close-range growth: while global affine scale is
        // explicitly propagating size, gradually forget older target-only rates.
        widthLogRate_ *= 0.65;
        heightLogRate_ *= 0.65;

        activeConfidence_ *= isolated_h_detail::MISS_CONFIDENCE_DECAY;
        activePeakConfidence_ = std::max(
            activeConfidence_,
            activePeakConfidence_ * isolated_h_detail::ACTIVE_PEAK_DECAY);

        state_ = State::RECOVER;
        observedThisFrame_ = false;
        predictedThisFrame_ = true;
        guidanceValid_ = true;
        searchMode_ = "PREDICT+FRAME-MOTION";

        recoveryTrackerConsecutive_ = 0;

        // This path is only entered after the normal multi-frame H acquisition
        // has already committed a real lock, so it receives the full bounded
        // verified-lock continuation allowance.
        if (misses_ >= isolated_h_detail::VERIFIED_LOCK_RESET_MISSES) {
            reset();
            return false;
        }

        return true;
    }

    std::optional<Candidate> reverifyRecoveryCandidate(
        const cv::Mat& gray,
        const Candidate& candidate,
        bool farTrack)
    {
        if (!corelock03_policy::requiresRecoveryReverify(
                farTrack, candidate.temporalSupported))
        {
            return candidate;
        }

        const cv::Size2f verifySize(
            std::max(6.0f, candidate.box.width),
            std::max(6.0f, candidate.box.height));
        DetectionOutput verify = detector_.trackKnownTarget(
            gray,
            candidate.center,
            verifySize,
            true);

        if (!verify.bestPatch.empty()) {
            acceptDebug(verify);
        }

        std::optional<Candidate> match =
            chooseActiveMatch(verify.candidates, candidate.center);
        if (match) {
            searchMode_ += "+HREVERIFY";
            return match;
        }

        searchMode_ += "+HREVERIFY-MISS";
        return std::nullopt;
    }

    void clearEscape()
    {
        escapeActive_ = false;
        escapeCandidate_ = {};
        escapeConfirmCount_ = 0;
        escapeRelocationReady_ = false;
    }

    std::optional<Candidate> considerVerifiedEscape(
        const cv::Mat& gray,
        const std::vector<Candidate>& candidates)
    {
        if (!active_ ||
            !corelock04_policy::escapeArmed(
                directHMissStreak_,
                isolated_h_detail::ESCAPE_MIN_DIRECT_MISSES))
        {
            clearEscape();
            return std::nullopt;
        }

        std::optional<Candidate> strongest;
        for (const Candidate& candidate : candidates) {
            if (!candidate.hVerified || !candidate.strictMode) {
                continue;
            }

            const double independentDistance = cv::norm(
                candidate.center - activeCenter_);
            const double independentFloor = std::max(
                18.0,
                0.75 * isolated_h_detail::rectMajor(activeBox_));
            if (independentDistance < independentFloor) {
                continue;
            }

            if (!strongest || candidate.finalScore > strongest->finalScore) {
                strongest = candidate;
            }
        }

        if (!strongest) {
            return std::nullopt;
        }

        const cv::Size2f verifySize(
            std::max(6.0f, strongest->box.width),
            std::max(6.0f, strongest->box.height));
        DetectionOutput verify = detector_.trackKnownTarget(
            gray,
            strongest->center,
            verifySize,
            true);

        if (!verify.bestPatch.empty()) {
            acceptDebug(verify);
        }
        if (verify.candidates.empty()) {
            return std::nullopt;
        }

        const Candidate verified = verify.candidates.front();
        const bool independentlyVerified =
            verified.hVerified && verified.strictMode;
        const bool resolvedIntegrityPass = verified.strictMode;
        if (!independentlyVerified || !resolvedIntegrityPass) {
            return std::nullopt;
        }

        if (escapeActive_ && sameIdentity(verified, escapeCandidate_)) {
            ++escapeConfirmCount_;
            escapeCandidate_ = verified;
        }
        else {
            escapeActive_ = true;
            escapeCandidate_ = verified;
            escapeConfirmCount_ = 1;
        }

        if (corelock04_policy::escapeReady(
                directHMissStreak_,
                independentlyVerified,
                resolvedIntegrityPass,
                escapeConfirmCount_,
                isolated_h_detail::ESCAPE_MIN_DIRECT_MISSES,
                isolated_h_detail::ESCAPE_CONFIRM_CHECKS))
        {
            escapeRelocationReady_ = true;
            searchMode_ += "+ESCAPE-COMMIT";
            return verified;
        }

        searchMode_ += "+ESCAPE-" + std::to_string(escapeConfirmCount_) +
            "/" + std::to_string(isolated_h_detail::ESCAPE_CONFIRM_CHECKS);
        return std::nullopt;
    }

    void relocateActiveVerified(
        const cv::Mat& gray,
        const Candidate& candidate)
    {
        active_ = true;
        activeProvisional_ = false;
        provisionalFreshCompatible_ = 0;
        provisionalMissStreak_ = 0;
        provisionalReference_ = {};
        state_ = State::TRACK;
        activeCenter_ = candidate.center;
        activeVelocity_ = cv::Point2f(0.0f, 0.0f);
        activeBox_ = candidate.box;
        activeEllipse_ = candidate.ellipseFull;
        activeEllipse_.center = activeCenter_;
        widthLogRate_ = 0.0;
        heightLogRate_ = 0.0;
        activeConfidence_ = std::max(
            candidate.finalScore,
            0.75 * activeConfidence_);
        activePeakConfidence_ = std::max(
            activePeakConfidence_, activeConfidence_);
        lastCandidate_ = candidate;
        misses_ = 0;
        directHMissStreak_ = 0;
        strongTrackStreak_ = 0;
        ++activeVerifiedFrames_;
        observedThisFrame_ = true;
        predictedThisFrame_ = false;
        guidanceValid_ = true;
        recoveryTrackerConsecutive_ = 0;
        recoveryTrackerSeedAge_ = 0;
        recoveryStageFrame_ = 0;
        frameMotionConsecutive_ = 0;
        seedRecoveryTracker(gray, candidate, true);
        clearEscape();
        searchMode_ += "+RELOCATED";
    }

    std::optional<Candidate> runPredictedScheduledSearch(
        const cv::Mat& gray,
        SearchTask task,
        double expectedMajor,
        bool farTrack,
        std::vector<Candidate>& decisionCandidates)
    {
        if (!active_ || task == SearchTask::SMALL) {
            return std::nullopt;
        }

        const cv::Point2f associationCenter = activeCenter_;
        DetectionOutput scan;

        if (task == SearchTask::MEDIUM) {
            const cv::Rect roi = targetCenteredRoi(gray, SearchTask::MEDIUM);
            activeTiles_.push_back(roi);

            if (farTrack) {
                cv::Mat fused = buildTemporalRoi(gray, roi);
                if (!fused.empty()) {
                    const cv::Rect whole(0, 0, fused.cols, fused.rows);
                    scan = detector_.detect(
                        fused, whole, false, true, expectedMajor, true);
                    offsetDetectionToFull(
                        scan,
                        cv::Point2f(
                            static_cast<float>(roi.x),
                            static_cast<float>(roi.y)));
                    for (Candidate& c : scan.candidates)
                        c.temporalSupported = true;
                    searchMode_ = "PREDICT+MEDIUM-TEMP";
                }
                else {
                    scan = detector_.detect(
                        gray, roi, false, true, expectedMajor);
                    searchMode_ = "PREDICT+MEDIUM";
                }
            }
            else {
                scan = detector_.detect(
                    gray, roi, false, true, expectedMajor);
                searchMode_ = "PREDICT+MEDIUM";
            }
        }
        else {
            const cv::Rect whole(0, 0, gray.cols, gray.rows);
            scan = detector_.detect(gray, whole, true, false);
            lastGlobalCandidateCount_ = static_cast<int>(scan.candidates.size());
            searchMode_ = "PREDICT+GLOBAL";
        }

        if (!scan.bestPatch.empty()) acceptDebug(scan);
        decisionCandidates.insert(
            decisionCandidates.end(),
            scan.candidates.begin(),
            scan.candidates.end());
        lastCandidates_ = scan.candidates;

        std::optional<Candidate> match =
            chooseActiveMatch(scan.candidates, associationCenter);
        if (match) {
            std::optional<Candidate> verified =
                reverifyRecoveryCandidate(gray, *match, farTrack);
            if (verified) return verified;
        }
        return considerVerifiedEscape(gray, scan.candidates);
    }

    std::optional<Candidate> runOneRecoveryStage(
        const cv::Mat& gray,
        double expectedMajor,
        bool farTrack,
        bool trackerMoved,
        std::vector<Candidate>& decisionCandidates)
    {
        if (!active_) return std::nullopt;

        const int recoveryFrame = std::max(1, recoveryStageFrame_);
        if (recoveryFrame == 1 && trackerMoved) {
            // First miss stays very cheap when the tracker successfully relocates
            // the target: tracker + corrected-location H reverify only.
            if (searchMode_.find("RECOVER") == std::string::npos)
                searchMode_ = "RECOVER+FAST";
            return std::nullopt;
        }

        int slot = 0;
        if (recoveryFrame == 1) {
            // If tracker relocation itself failed (common for a still-tiny H),
            // spend exactly one LOCAL recovery pass immediately rather than
            // waiting a frame with no useful fallback.
            slot = 2;
        }
        else if (recoveryFrame <= 4) {
            slot = recoveryFrame; // 2=LOCAL, 3=CENTER, 4=GLOBAL
        }
        else {
            slot = 2 + ((recoveryFrame - 5) % 3); // 2,3,4 repeating
        }

        const cv::Point2f associationCenter = activeCenter_;

        if (slot == 2) {
            const cv::Rect roi = targetCenteredRoi(gray, SearchTask::SMALL);
            activeTiles_.push_back(roi);
            DetectionOutput recovery;

            if (farTrack) {
                cv::Mat fused = buildTemporalRoi(gray, roi);
                if (!fused.empty()) {
                    const cv::Rect whole(0, 0, fused.cols, fused.rows);
                    recovery = detector_.detect(
                        fused, whole, false, true, expectedMajor, true);
                    offsetDetectionToFull(
                        recovery,
                        cv::Point2f(
                            static_cast<float>(roi.x),
                            static_cast<float>(roi.y)));
                    for (Candidate& c : recovery.candidates)
                        c.temporalSupported = true;
                    searchMode_ = "RECOVER+LOCAL-TEMP";
                }
                else {
                    recovery = detector_.detect(
                        gray, roi, false, true, expectedMajor);
                    searchMode_ = "RECOVER+LOCAL";
                }
            }
            else {
                recovery = detector_.detect(
                    gray, roi, false, true, expectedMajor);
                searchMode_ = "RECOVER+LOCAL";
            }

            if (!recovery.bestPatch.empty()) acceptDebug(recovery);
            decisionCandidates.insert(
                decisionCandidates.end(),
                recovery.candidates.begin(),
                recovery.candidates.end());
            lastCandidates_ = recovery.candidates;
            std::optional<Candidate> match =
                chooseActiveMatch(recovery.candidates, associationCenter);
            if (!match) return std::nullopt;
            return reverifyRecoveryCandidate(gray, *match, farTrack);
        }

        if (slot == 3) {
            const cv::Rect centreRoi = nextGridRoi(gray.size(), false);
            activeTiles_.push_back(centreRoi);
            DetectionOutput centre = detector_.detect(
                gray, centreRoi, false, false);
            if (!centre.bestPatch.empty()) acceptDebug(centre);
            decisionCandidates.insert(
                decisionCandidates.end(),
                centre.candidates.begin(),
                centre.candidates.end());
            lastCandidates_ = centre.candidates;
            searchMode_ = "RECOVER+CENTER";
            std::optional<Candidate> match =
                chooseActiveMatch(centre.candidates, associationCenter);
            if (match) {
                std::optional<Candidate> verified =
                    reverifyRecoveryCandidate(gray, *match, farTrack);
                if (verified) return verified;
            }
            return considerVerifiedEscape(gray, centre.candidates);
        }

        const cv::Rect whole(0, 0, gray.cols, gray.rows);
        DetectionOutput global = detector_.detect(
            gray, whole, true, false);
        lastGlobalCandidateCount_ = static_cast<int>(global.candidates.size());
        if (!global.bestPatch.empty()) acceptDebug(global);
        decisionCandidates.insert(
            decisionCandidates.end(),
            global.candidates.begin(),
            global.candidates.end());
        lastCandidates_ = global.candidates;
        searchMode_ = "RECOVER+GLOBAL";
        std::optional<Candidate> match =
            chooseActiveMatch(global.candidates, associationCenter);
        if (match) {
            std::optional<Candidate> verified =
                reverifyRecoveryCandidate(gray, *match, farTrack);
            if (verified) return verified;
        }
        return considerVerifiedEscape(gray, global.candidates);
    }

    std::optional<Candidate> chooseActiveMatch(
        const std::vector<Candidate>& candidates,
        const cv::Point2f& predicted) const
    {
        using namespace isolated_h_detail;

        std::optional<Candidate> best;
        double bestCost = std::numeric_limits<double>::infinity();

        const cv::Size2f predictedSize = predictedTargetSize();
        const double predictedW = std::max(6.0, static_cast<double>(predictedSize.width));
        const double predictedH = std::max(6.0, static_cast<double>(predictedSize.height));
        const double predictedDiag = std::max(10.0, std::hypot(predictedW, predictedH));
        const double predictedAspect = predictedW / predictedH;
        const double minRatio = 1.0 / MATCH_MAX_SHRINK_RATIO;

        for (const Candidate& candidate : candidates) {
            const double candidateW = std::max(4.0, static_cast<double>(candidate.box.width));
            const double candidateH = std::max(4.0, static_cast<double>(candidate.box.height));
            const double wr = candidateW / predictedW;
            const double hr = candidateH / predictedH;

            if (wr < minRatio || hr < minRatio ||
                wr > MATCH_MAX_GROW_RATIO || hr > MATCH_MAX_GROW_RATIO)
            {
                continue;
            }

            const double candidateAspect = candidateW / candidateH;
            const double aspectChange = std::max(
                candidateAspect / std::max(1e-6, predictedAspect),
                predictedAspect / std::max(1e-6, candidateAspect));

            if (aspectChange > MATCH_MAX_ASPECT_CHANGE) {
                continue;
            }

            const double distance = cv::norm(candidate.center - predicted);
            const double gate = std::max(
                18.0,
                0.70 * predictedDiag +
                2.5 * cv::norm(activeVelocity_) +
                5.0 * misses_);

            if (distance > gate) {
                continue;
            }

            const double sizeCost =
                std::abs(std::log(std::max(1e-6, wr))) +
                std::abs(std::log(std::max(1e-6, hr)));
            const double aspectCost = std::abs(std::log(std::max(
                1e-6, candidateAspect / predictedAspect)));

            const double cost =
                distance / predictedDiag +
                0.28 * sizeCost +
                0.16 * aspectCost -
                0.68 * candidate.finalScore;

            if (cost < bestCost) {
                bestCost = cost;
                best = candidate;
            }
        }

        return best;
    }

    bool provisionalCandidateCompatible(const Candidate& candidate) const
    {
        if (!activeProvisional_) return true;
        if (!candidate.hVerified) return false;
        if (!candidate.farMode) return candidate.strictMode;
        if (candidate.farSignatureKind == corelock05_policy::FarSignatureKind::Reject)
            return false;
        if (!provisionalReference_.farMode ||
            provisionalReference_.farSignatureKind ==
                corelock05_policy::FarSignatureKind::Reject)
        {
            return true;
        }
        return corelock05_policy::farSignaturesCompatible(
            candidateFarTopology(provisionalReference_),
            provisionalReference_.darkPolarity,
            provisionalReference_.ellipseFull.angle,
            candidateFarTopology(candidate),
            candidate.darkPolarity,
            candidate.ellipseFull.angle);
    }

    void lockCandidate(const cv::Mat& gray, const Candidate& candidate)
    {
        active_ = true;
        state_ = State::TRACK;
        activeCenter_ = candidate.center;
        activeVelocity_ = cv::Point2f(0.0f, 0.0f);
        activeBox_ = candidate.box;
        activeEllipse_ = candidate.ellipseFull;
        widthLogRate_ = 0.0;
        heightLogRate_ = 0.0;
        activeConfidence_ = candidate.finalScore;
        activePeakConfidence_ = candidate.finalScore;
        lastCandidate_ = candidate;
        activeProvisional_ = corelock05_policy::shouldStartProvisional(
            candidate.farMode, candidate.farSignatureKind);
        provisionalFreshCompatible_ = 0;
        provisionalMissStreak_ = 0;
        provisionalReference_ = candidate;
        if (activeProvisional_) searchMode_ += "+PROVISIONAL-H";
        misses_ = 0;
        directHMissStreak_ = 0;
        strongTrackStreak_ = 0;
        activeVerifiedFrames_ = 1;
        observedThisFrame_ = true;
        predictedThisFrame_ = false;
        guidanceValid_ = true;
        clearPending();
        clearChallenger();
        recoveryTrackerConsecutive_ = 0;
        recoveryTrackerSeedAge_ = 0;
        recoveryStageFrame_ = 0;
        frameMotionConsecutive_ = 0;
        clearEscape();
        if (activeProvisional_) {
            recoveryTracker_.reset();
            recoveryTrackerSeedAge_ = 0;
        }
        else {
            seedRecoveryTracker(gray, candidate, true);
        }
    }

    void updateActive(const cv::Mat& gray, const Candidate& candidate)
    {
        if (activeProvisional_) {
            const bool compatible = provisionalCandidateCompatible(candidate);
            const bool freshCompatible = compatible && !candidate.temporalSupported;
            if (freshCompatible) ++provisionalFreshCompatible_;
            else if (!compatible) provisionalFreshCompatible_ = 0;
            provisionalMissStreak_ = 0;
            provisionalReference_ = candidate;

            if (corelock05_policy::provisionalPromoted(
                    provisionalFreshCompatible_,
                    isolated_h_detail::PROVISIONAL_PROMOTE_FRESH))
            {
                activeProvisional_ = false;
                searchMode_ += "+TRUSTED-H";
            }
            else {
                searchMode_ += "+PROVISIONAL-H";
            }
        }

        const cv::Point2f predicted =
            motionCompensatedPoint(activeCenter_) + activeVelocity_;

        cv::Point2f residual = candidate.center - predicted;

        const double activeSize = std::max(
            10.0, isolated_h_detail::rectMajor(activeBox_));
        const double maxResidual = std::max(6.0, 0.75 * activeSize);
        const double residualNorm = cv::norm(residual);
        if (residualNorm > maxResidual && residualNorm > 1e-6) {
            residual *= static_cast<float>(maxResidual / residualNorm);
        }

        const cv::Point2f corrected =
            predicted + residual * static_cast<float>(isolated_h_detail::POSITION_ALPHA);

        activeVelocity_ =
            activeVelocity_ * static_cast<float>(1.0 - isolated_h_detail::VELOCITY_BETA) +
            (corrected - activeCenter_) * static_cast<float>(isolated_h_detail::VELOCITY_BETA);

        const cv::Rect2f oldBox = activeBox_;
        activeCenter_ = corrected;

        const double oldW = std::max(4.0, static_cast<double>(oldBox.width));
        const double oldH = std::max(4.0, static_cast<double>(oldBox.height));
        const double newW = std::max(4.0, static_cast<double>(candidate.box.width));
        const double newH = std::max(4.0, static_cast<double>(candidate.box.height));

        const double observedWRate = isolated_h_detail::clampv(
            std::log(newW / oldW),
            isolated_h_detail::MAX_LOG_SHRINK_PER_FRAME,
            isolated_h_detail::MAX_LOG_GROWTH_PER_FRAME);
        const double observedHRate = isolated_h_detail::clampv(
            std::log(newH / oldH),
            isolated_h_detail::MAX_LOG_SHRINK_PER_FRAME,
            isolated_h_detail::MAX_LOG_GROWTH_PER_FRAME);

        widthLogRate_ =
            (1.0 - isolated_h_detail::SIZE_RATE_ALPHA) * widthLogRate_ +
            isolated_h_detail::SIZE_RATE_ALPHA * observedWRate;
        heightLogRate_ =
            (1.0 - isolated_h_detail::SIZE_RATE_ALPHA) * heightLogRate_ +
            isolated_h_detail::SIZE_RATE_ALPHA * observedHRate;

        // Fast growth / slow shrink: a rapidly approaching H expands the tracking
        // box quickly, while a partial or blurred observation cannot collapse it.
        const double widthAlpha = newW >= oldW
            ? isolated_h_detail::WIDTH_HEIGHT_GROW_ALPHA
            : isolated_h_detail::WIDTH_HEIGHT_SHRINK_ALPHA;
        const double heightAlpha = newH >= oldH
            ? isolated_h_detail::WIDTH_HEIGHT_GROW_ALPHA
            : isolated_h_detail::WIDTH_HEIGHT_SHRINK_ALPHA;

        activeBox_.width = static_cast<float>(
            (1.0 - widthAlpha) * oldW + widthAlpha * newW);
        activeBox_.height = static_cast<float>(
            (1.0 - heightAlpha) * oldH + heightAlpha * newH);

        activeBox_.x = activeCenter_.x - 0.5f * activeBox_.width;
        activeBox_.y = activeCenter_.y - 0.5f * activeBox_.height;

        activeEllipse_ = candidate.ellipseFull;
        activeEllipse_.center = activeCenter_;

        activeConfidence_ =
            0.55 * activeConfidence_ +
            0.45 * candidate.finalScore;

        activePeakConfidence_ = std::max(
            activeConfidence_,
            activePeakConfidence_ * isolated_h_detail::ACTIVE_PEAK_DECAY);

        lastCandidate_ = candidate;
        lastCandidate_.center = activeCenter_;
        lastCandidate_.box = activeBox_;
        lastCandidate_.ellipseFull = activeEllipse_;
        lastCandidate_.finalScore = activeConfidence_;

        misses_ = 0;
        directHMissStreak_ = 0;
        ++activeVerifiedFrames_;
        state_ = State::TRACK;
        observedThisFrame_ = true;
        predictedThisFrame_ = false;
        guidanceValid_ = true;

        recoveryTrackerConsecutive_ = 0;
        recoveryStageFrame_ = 0;
        frameMotionConsecutive_ = 0;
        ++recoveryTrackerSeedAge_;
        clearEscape();
        if (activeProvisional_) {
            recoveryTracker_.reset();
            recoveryTrackerSeedAge_ = 0;
        }
        else {
            seedRecoveryTracker(gray, candidate, false);
        }
    }

    void seedRecoveryTracker(
        const cv::Mat& gray,
        const Candidate& candidate,
        bool force)
    {
        if (gray.empty() ||
            isolated_h_detail::rectMajor(candidate.box) <
                isolated_h_detail::RECOVERY_TRACKER_MIN_MAJOR_PX)
        {
            recoveryTracker_.reset();
            recoveryTrackerSeedAge_ = 0;
            return;
        }

        if (!force && recoveryTracker_.isActive() &&
            recoveryTrackerSeedAge_ <
                isolated_h_detail::RECOVERY_TRACKER_RESEED_EVERY_VERIFIED)
        {
            return;
        }

        cv::Rect box(
            static_cast<int>(std::floor(candidate.box.x)),
            static_cast<int>(std::floor(candidate.box.y)),
            std::max(1, static_cast<int>(std::ceil(candidate.box.width))),
            std::max(1, static_cast<int>(std::ceil(candidate.box.height))));
        box &= cv::Rect(0, 0, gray.cols, gray.rows);

        if (box.area() <= 0 || !recoveryTracker_.initialize(gray, box)) {
            recoveryTracker_.reset();
        }

        recoveryTrackerSeedAge_ = 0;
    }

    bool applyRecoveryTracker(
        const cv::Mat& gray)
    {
        if (!recoveryTracker_.isActive() ||
            recoveryTrackerConsecutive_ >=
                isolated_h_detail::RECOVERY_TRACKER_MAX_CONSECUTIVE)
        {
            return false;
        }

        cv::Rect recoveredTarget;
        cv::Rect recoveredAnchor;
        if (!recoveryTracker_.update(gray, recoveredTarget, recoveredAnchor) ||
            recoveredTarget.area() <= 0 ||
            recoveryTracker_.targetConfidence() <
                isolated_h_detail::RECOVERY_TRACKER_MIN_CONF)
        {
            return false;
        }

        ++misses_;
        ++recoveryTrackerConsecutive_;
        ++recoveryStageFrame_;

        const cv::Point2f previousCenter = activeCenter_;
        const cv::Point2f recoveredCenter(
            recoveredTarget.x + 0.5f * recoveredTarget.width,
            recoveredTarget.y + 0.5f * recoveredTarget.height);

        const cv::Point2f recoveredDelta = recoveredCenter - previousCenter;
        activeVelocity_ =
            0.72f * activeVelocity_ + 0.28f * recoveredDelta;
        activeCenter_ = recoveredCenter;

        // Do not pretend a template match is a fresh H observation. Preserve the
        // last H geometry and only move the recovery window/center.
        activeBox_.x = activeCenter_.x - 0.5f * activeBox_.width;
        activeBox_.y = activeCenter_.y - 0.5f * activeBox_.height;
        activeEllipse_.center = activeCenter_;

        activeConfidence_ *= isolated_h_detail::MISS_CONFIDENCE_DECAY;
        activePeakConfidence_ = std::max(
            activeConfidence_,
            activePeakConfidence_ * isolated_h_detail::ACTIVE_PEAK_DECAY);

        state_ = State::RECOVER;
        observedThisFrame_ = false;
        predictedThisFrame_ =
            recoveryTrackerConsecutive_ <=
                isolated_h_detail::RECOVERY_TRACKER_GUIDANCE_FRAMES &&
            activeConfidence_ >= isolated_h_detail::MIN_PREDICT_CONFIDENCE;
        guidanceValid_ = predictedThisFrame_;
        searchMode_ = "RECOVER+USER-TRACKER";

        const int resetMissThreshold =
            activeVerifiedFrames_ >=
                isolated_h_detail::VERIFIED_LOCK_FRAMES_FOR_LONG_RECOVERY
            ? isolated_h_detail::VERIFIED_LOCK_RESET_MISSES
            : isolated_h_detail::NEW_LOCK_RESET_MISSES;

        if (misses_ >= resetMissThreshold) {
            reset();
            return false;
        }

        return true;
    }

    void handleLocalMiss(
        const cv::Mat& gray,
        const cv::Point2f& predictedCenter)
    {
        ++misses_;
        ++recoveryStageFrame_;

        // Keep the recovery window attached to the same ground point while the
        // camera wobbles. This is the already-computed motion-compensated point,
        // so the frame transform is applied exactly once per input frame.
        activeCenter_ = predictedCenter;
        activeVelocity_ *= 0.65f;
        activeBox_.x = activeCenter_.x - 0.5f * activeBox_.width;
        activeBox_.y = activeCenter_.y - 0.5f * activeBox_.height;
        activeEllipse_.center = activeCenter_;

        activeConfidence_ *= isolated_h_detail::MISS_CONFIDENCE_DECAY;
        activePeakConfidence_ = std::max(
            activeConfidence_,
            activePeakConfidence_ * isolated_h_detail::ACTIVE_PEAK_DECAY);

        state_ = State::RECOVER;
        outputShortPrediction(gray);

        const int resetMissThreshold =
            activeVerifiedFrames_ >= isolated_h_detail::VERIFIED_LOCK_FRAMES_FOR_LONG_RECOVERY
            ? isolated_h_detail::VERIFIED_LOCK_RESET_MISSES
            : isolated_h_detail::NEW_LOCK_RESET_MISSES;

        if (misses_ >= resetMissThreshold) {
            reset();
        }
    }

    void outputShortPrediction(const cv::Mat& gray)
    {
        (void)gray;
        // Never advance the committed coordinate on a miss. Hold the LAST
        // VERIFIED coordinate for at most two frames to avoid display/control
        // flicker, then withhold guidance while local recovery continues.
        if (misses_ <= isolated_h_detail::HOLD_LAST_GUIDANCE_FRAMES &&
            activeConfidence_ >= isolated_h_detail::MIN_PREDICT_CONFIDENCE)
        {
            predictedThisFrame_ = true;
            guidanceValid_ = true;
        }
        else {
            predictedThisFrame_ = false;
            guidanceValid_ = false;
        }
    }

    void processChallenger(
        const std::vector<Candidate>& candidates,
        const std::optional<Candidate>& activeMatch)
    {
        if (!active_) {
            return;
        }

        const double drop = activePeakConfidence_ > 1e-6
            ? 1.0 - activeConfidence_ / activePeakConfidence_
            : 0.0;

        const bool hardLost =
            misses_ >= isolated_h_detail::HARD_LOST_SWITCH_MISSES;

        if (!hardLost &&
            drop < isolated_h_detail::SWITCH_DROP_FRACTION)
        {
            clearChallenger();
            return;
        }

        std::optional<Candidate> strongest;

        for (const Candidate& candidate : candidates) {
            if (activeMatch && sameIdentity(candidate, *activeMatch)) {
                continue;
            }

            const double requiredAdvantage = hardLost
                ? 0.05
                : isolated_h_detail::SWITCH_MIN_ADVANTAGE;

            if (candidate.finalScore <
                activeConfidence_ + requiredAdvantage)
            {
                continue;
            }

            if (!strongest ||
                candidate.finalScore > strongest->finalScore)
            {
                strongest = candidate;
            }
        }

        if (!strongest) {
            clearChallenger();
            return;
        }

        if (challengerActive_ &&
            sameIdentity(*strongest, challengerCandidate_))
        {
            ++challengerCount_;
            challengerCandidate_ = *strongest;
        }
        else {
            challengerActive_ = true;
            challengerCandidate_ = *strongest;
            challengerCount_ = 1;
        }

        if (challengerCount_ >= isolated_h_detail::SWITCH_CONFIRM_CHECKS) {
            lockCandidate(cv::Mat(), challengerCandidate_);
        }
    }

    static bool sameIdentity(
        const Candidate& a,
        const Candidate& b)
    {
        const double size = std::max(
            isolated_h_detail::rectMajor(a.box),
            isolated_h_detail::rectMajor(b.box));

        const double distance = cv::norm(a.center - b.center);

        return distance < std::max(18.0, 1.25 * size);
    }

    void acceptDebug(const DetectionOutput& detection)
    {
        debugPatch_ = detection.bestPatch.clone();
        debugH_ = detection.bestHMask.clone();
    }

    void clearPending()
    {
        pendingActive_ = false;
        pendingCandidate_ = {};
        pendingCount_ = 0;
        pendingAge_ = 0;
        pendingTemporalSeen_ = false;
        pendingTemporalTopologySeen_ = false;
        pendingTopologyStrongCount_ = 0;
        pendingClutterCleanCount_ = 0;
        pendingContinuationCleanCount_ = 0;
        pendingFarSignatureCompatCount_ = 0;
        pendingFocusMisses_ = 0;
        focusFramesRemaining_ = 0;
    }

    void clearChallenger()
    {
        challengerActive_ = false;
        challengerCandidate_ = {};
        challengerCount_ = 0;
    }

    cv::Point2f motionCompensatedPoint(const cv::Point2f& p) const
    {
        if (!currentMotion_.valid) return p;
        return isolated_h_detail::transformPoint(
            currentMotion_.prevToCurrent, p);
    }

    struct TemporalFrame
    {
        cv::Mat gray;
        cv::Matx23f toCurrent{1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f};
    };

    void updateTemporalHistory(const cv::Mat& gray)
    {
        using namespace isolated_h_detail;

        if (currentMotion_.valid) {
            for (TemporalFrame& item : temporalHistory_) {
                item.toCurrent = composeAffine(
                    currentMotion_.prevToCurrent, item.toCurrent);
            }
        }
        else if (!temporalHistory_.empty()) {
            // Without a reliable frame-to-frame transform, old images are not
            // safe to fuse. Keep only the current frame.
            temporalHistory_.clear();
        }

        TemporalFrame current;
        // cv::Mat is reference-counted; retaining it avoids a full 1080p copy.
        current.gray = gray;
        temporalHistory_.push_front(std::move(current));
        while (static_cast<int>(temporalHistory_.size()) > TEMPORAL_HISTORY_FRAMES) {
            temporalHistory_.pop_back();
        }
    }

    cv::Mat buildTemporalRoi(const cv::Mat& gray, const cv::Rect& roi) const
    {
        using namespace isolated_h_detail;

        if (static_cast<int>(temporalHistory_.size()) < TEMPORAL_MIN_FRAMES ||
            roi.width < 8 || roi.height < 8)
        {
            return {};
        }

        cv::Mat accumulator = cv::Mat::zeros(roi.size(), CV_32F);
        double weightSum = 0.0;

        // Before a clear lock we deliberately keep all available aligned history,
        // up to the full five-frame buffer, to recover H legs seen on alternating
        // far-range frames.
        const int count = std::min(
            static_cast<int>(temporalHistory_.size()),
            TEMPORAL_HISTORY_FRAMES);

        for (int i = 0; i < count; ++i) {
            const TemporalFrame& item = temporalHistory_[static_cast<size_t>(i)];
            cv::Mat patch;

            if (i == 0) {
                patch = gray(roi);
            }
            else {
                cv::Matx23f m = item.toCurrent;
                m(0,2) -= static_cast<float>(roi.x);
                m(1,2) -= static_cast<float>(roi.y);
                cv::warpAffine(
                    item.gray, patch, cv::Mat(m), roi.size(),
                    cv::INTER_LINEAR, cv::BORDER_REFLECT101);
            }

            cv::Mat f;
            patch.convertTo(f, CV_32F);
            const double w = TEMPORAL_WEIGHTS[static_cast<size_t>(i)];
            accumulator += f * w;
            weightSum += w;
        }

        if (weightSum <= 1e-6) return {};
        accumulator *= (1.0 / weightSum);

        cv::Mat fused;
        accumulator.convertTo(fused, CV_8U);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.6, cv::Size(8,8));
        clahe->apply(fused, fused);
        return fused;
    }

    static void offsetDetectionToFull(
        DetectionOutput& detection,
        const cv::Point2f& offset)
    {
        for (Candidate& c : detection.candidates) {
            c.center += offset;
            c.box.x += offset.x;
            c.box.y += offset.y;
            c.identityBox.x += offset.x;
            c.identityBox.y += offset.y;
            c.supportBox.x += offset.x;
            c.supportBox.y += offset.y;
            c.ellipseFull.center += offset;
        }
    }

    void reset()
    {
        active_ = false;
        state_ = State::SEARCH;
        activeCenter_ = cv::Point2f(0.0f, 0.0f);
        activeVelocity_ = cv::Point2f(0.0f, 0.0f);
        activeBox_ = {};
        activeEllipse_ = {};
        widthLogRate_ = 0.0;
        heightLogRate_ = 0.0;
        activeConfidence_ = 0.0;
        activePeakConfidence_ = 0.0;
        lastCandidate_ = {};
        misses_ = 0;
        directHMissStreak_ = 0;
        strongTrackStreak_ = 0;
        activeVerifiedFrames_ = 0;
        activeProvisional_ = false;
        provisionalFreshCompatible_ = 0;
        provisionalMissStreak_ = 0;
        provisionalReference_ = {};
        pendingFocusMisses_ = 0;
        observedThisFrame_ = false;
        predictedThisFrame_ = false;
        guidanceValid_ = false;
        recoveryTracker_.reset();
        recoveryTrackerSeedAge_ = 0;
        recoveryTrackerConsecutive_ = 0;
        recoveryStageFrame_ = 0;
        frameMotionConsecutive_ = 0;
        clearPending();
        clearChallenger();
        clearEscape();
    }

private:
    isolated_h_detail::IsolatedHDetector detector_;
    StrictTracker recoveryTracker_;
    int recoveryTrackerSeedAge_ = 0;
    int recoveryTrackerConsecutive_ = 0;
    int recoveryStageFrame_ = 0;
    int frameMotionConsecutive_ = 0;

    State state_ = State::SEARCH;
    bool active_ = false;

    cv::Point2f activeCenter_{0.0f, 0.0f};
    cv::Point2f activeVelocity_{0.0f, 0.0f};
    cv::Rect2f activeBox_;
    cv::RotatedRect activeEllipse_;

    // Independent scale rates allow the ROI to change aspect ratio as the
    // projected H changes under perspective while the aircraft approaches.
    double widthLogRate_ = 0.0;
    double heightLogRate_ = 0.0;

    Candidate lastCandidate_;

    double activeConfidence_ = 0.0;
    double activePeakConfidence_ = 0.0;

    int frameId_ = 0;
    int scheduleIndex_ = 0;
    int smallGridIndex_ = 0;
    int mediumGridIndex_ = 0;
    int misses_ = 0;
    int directHMissStreak_ = 0;

    bool observedThisFrame_ = false;
    bool predictedThisFrame_ = false;
    bool guidanceValid_ = false;

    bool pendingActive_ = false;
    Candidate pendingCandidate_;
    int pendingCount_ = 0;
    int pendingAge_ = 0;
    bool pendingTemporalSeen_ = false;
    bool pendingTemporalTopologySeen_ = false;
    int pendingTopologyStrongCount_ = 0;
    int pendingClutterCleanCount_ = 0;
    int pendingContinuationCleanCount_ = 0;
    int pendingFarSignatureCompatCount_ = 0;
    int focusFramesRemaining_ = 0;

    // Consecutive resolved-size strong observations required before the cheap
    // STICKY-FAST path is allowed. Tiny FAR_TRACK never uses that early exit.
    int strongTrackStreak_ = 0;
    int activeVerifiedFrames_ = 0;
    bool activeProvisional_ = false;
    int provisionalFreshCompatible_ = 0;
    int provisionalMissStreak_ = 0;
    Candidate provisionalReference_;
    int pendingFocusMisses_ = 0;

    bool challengerActive_ = false;
    Candidate challengerCandidate_;
    int challengerCount_ = 0;

    bool escapeActive_ = false;
    Candidate escapeCandidate_;
    int escapeConfirmCount_ = 0;
    bool escapeRelocationReady_ = false;

    int lastGlobalCandidateCount_ = 0;
    std::string searchMode_ = "GLOBAL";
    bool externalCandidateOfferedSinceUpdate_ = false;

    std::vector<Candidate> lastCandidates_;
    std::vector<cv::Rect> activeTiles_;
    cv::Mat debugPatch_;
    cv::Mat debugH_;

    isolated_h_detail::FrameMotion currentMotion_;
    std::deque<TemporalFrame> temporalHistory_;
};

// ============================================================
// Generic modular interface used by the loop
// ============================================================

struct DetectedObject
{
    std::string className;
    cv::Point2f center;
    cv::Rect2f box;
    cv::RotatedRect markerEllipse;
    double confidence = 0.0;
};

struct CameraAttitude
{
    bool valid = false;
    double rollDeg = 0.0;
    double pitchDeg = 0.0;
    double yawDeg = 0.0;
    double ageMs = 0.0;
};

struct LogicOutput
{
    std::vector<DetectedObject> objects;
    SearchTrack::Output tracker;
    CameraAttitude attitude;
    isolated_h_detail::FrameMotion frameMotion;
    bool attitudeGateOpen = true;
};

class DetectionLogic
{
public:
    explicit DetectionLogic(int reverifyEvery)
        : tracker_(reverifyEvery)
    {
    }

    LogicOutput process(
        const cv::Mat& frame,
        const CameraAttitude& attitude = {})
    {
        LogicOutput output;
        output.attitude = attitude;

        if (frame.empty()) {
            return output;
        }

        cv::Mat gray;

        if (frame.channels() == 1) {
            gray = frame;
        }
        else if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }
        else if (frame.channels() == 4) {
            cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
        }
        else {
            return output;
        }

        ++motionFrameCounter_;
        const bool highRateMotion = tracker_.needsHighRateMotion();
        const bool refreshMotion = highRateMotion ||
            (motionFrameCounter_ % isolated_h_detail::MOTION_EVERY_N_FRAMES) == 1;
        output.frameMotion = motionEstimator_.update(gray, refreshMotion);

        bool detectionEnabled = true;
        if (attitude.valid && tracker_.state() == SearchTrack::State::SEARCH) {
            if (!attitudeWasValid_) {
                // First fresh FC sample: apply the START threshold directly.
                attitudeGateOpen_ =
                    attitude.pitchDeg < isolated_h_detail::CAMERA_PITCH_START_DEG;
            }
            else if (attitudeGateOpen_) {
                if (attitude.pitchDeg > isolated_h_detail::CAMERA_PITCH_STOP_DEG) {
                    attitudeGateOpen_ = false;
                }
            }
            else {
                if (attitude.pitchDeg < isolated_h_detail::CAMERA_PITCH_START_DEG) {
                    attitudeGateOpen_ = true;
                }
            }
            attitudeWasValid_ = true;
            detectionEnabled = attitudeGateOpen_;
        }
        else if (!attitude.valid) {
            // FC attitude is optional. Missing/stale data must never block vision.
            attitudeWasValid_ = false;
            detectionEnabled = true;
        }
        else {
            // Once locked, keep tracking through short attitude excursions.
            attitudeWasValid_ = true;
            detectionEnabled = true;
        }

        output.attitudeGateOpen = detectionEnabled;
        output.tracker = tracker_.update(gray, output.frameMotion, detectionEnabled);

        if (output.tracker.hasBox &&
            output.tracker.guidanceValid &&
            output.tracker.state != SearchTrack::State::SEARCH)
        {
            DetectedObject object;
            object.className = "H";
            object.center = output.tracker.center;
            object.box = output.tracker.box;
            object.markerEllipse = output.tracker.trackedEllipse;
            object.confidence = output.tracker.activeConfidence;
            output.objects.push_back(object);
        }

        return output;
    }

private:
    SearchTrack tracker_;
    isolated_h_detail::GlobalMotionEstimator motionEstimator_;
    uint64_t motionFrameCounter_ = 0;
    bool attitudeGateOpen_ = false;
    bool attitudeWasValid_ = false;
};

