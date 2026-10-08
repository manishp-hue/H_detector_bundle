#pragma once

#include <algorithm>
#include <array>
#include <cmath>

// V15.54 policy layer adapted from track_noml/PureH_CoreLock_05.
// This header is deliberately OpenCV-free so its decision thresholds can be
// unit-tested independently. Pixel measurements are produced in vision.cpp.
namespace helipad_corelock05 {

enum class FarSignatureKind {
    Strong = 0,
    Ambiguous = 1,
    Reject = 2
};

struct FarTopologyMetrics {
    double leftTop = 0.0;
    double leftBottom = 0.0;
    double rightTop = 0.0;
    double rightBottom = 0.0;
    double bridgeStrength = 0.0;
    double bridgeY = 0.5;
    double gapTop = 0.0;
    double gapBottom = 0.0;
    double leftRightBalance = 0.0;
    double blobCompactness = 1.0;
    double largestComponentFraction = 0.0;
    double twoLegRowContinuity = 0.0;
};

struct FarTopologyDecision {
    FarSignatureKind kind = FarSignatureKind::Reject;
    double score = 0.0;
};

inline double clamp01(double v) {
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

inline double min4(double a, double b, double c, double d) {
    return std::min(std::min(a, b), std::min(c, d));
}

// Thresholds and weights are intentionally kept byte-for-byte numerically
// equivalent to PureH_CoreLock_05/pure_h_corelock05_policy.hpp.
inline FarTopologyDecision classifyFarTopology(const FarTopologyMetrics& m) {
    FarTopologyDecision out;

    const double legMin = min4(
        m.leftTop, m.leftBottom, m.rightTop, m.rightBottom);
    const double gapMin = std::min(m.gapTop, m.gapBottom);
    const double bridgeInterior =
        (m.bridgeY >= 0.22 && m.bridgeY <= 0.78) ? 1.0 : 0.0;
    const double antiBlob = clamp01((0.74 - m.blobCompactness) / 0.42);

    out.score = clamp01(
        0.30 * legMin +
        0.22 * m.bridgeStrength +
        0.16 * gapMin +
        0.10 * m.leftRightBalance +
        0.08 * antiBlob +
        0.07 * m.largestComponentFraction +
        0.11 * m.twoLegRowContinuity +
        0.04 * bridgeInterior);

    const bool missingStemHalf =
        (m.leftTop < 0.10 && m.leftBottom < 0.10) ||
        (m.rightTop < 0.10 && m.rightBottom < 0.10) ||
        (m.leftTop < 0.08 && m.rightTop < 0.08) ||
        (m.leftBottom < 0.08 && m.rightBottom < 0.08);

    const bool compactBlob =
        m.blobCompactness >= 0.74 &&
        (gapMin < 0.18 || legMin < 0.22);

    const bool noCommonBridge =
        m.bridgeStrength < 0.13 ||
        m.bridgeY < 0.16 ||
        m.bridgeY > 0.84;

    const bool noCentralSeparation =
        m.gapTop < 0.08 && m.gapBottom < 0.08;

    if (missingStemHalf || compactBlob || noCommonBridge || noCentralSeparation) {
        out.kind = FarSignatureKind::Reject;
        return out;
    }

    const bool strong =
        legMin >= 0.43 &&
        m.bridgeStrength >= 0.46 &&
        gapMin >= 0.42 &&
        m.leftRightBalance >= 0.58 &&
        m.blobCompactness <= 0.62 &&
        m.largestComponentFraction >= 0.48 &&
        m.twoLegRowContinuity >= 0.34 &&
        bridgeInterior > 0.5 &&
        out.score >= 0.60;

    if (strong) {
        out.kind = FarSignatureKind::Strong;
        return out;
    }

    const bool plausible =
        legMin >= 0.13 &&
        m.bridgeStrength >= 0.18 &&
        gapMin >= 0.12 &&
        m.leftRightBalance >= 0.34 &&
        m.blobCompactness <= 0.76 &&
        m.largestComponentFraction >= 0.35 &&
        m.twoLegRowContinuity >= 0.22 &&
        m.bridgeY >= 0.18 &&
        m.bridgeY <= 0.82 &&
        out.score >= 0.28;

    out.kind = plausible ? FarSignatureKind::Ambiguous : FarSignatureKind::Reject;
    return out;
}

struct IntegrityMetrics {
    bool resolved = false;
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

struct IntegrityResult {
    bool pass = true;
    bool hardChecked = false;
    double score = 1.0;
};

static constexpr double MIN_SPANNING_COMPONENT = 0.50;
static constexpr double MIN_THREE_OF_FOUR_STEMS = 0.28;
static constexpr double MIN_BOTH_SIDES = 0.30;
static constexpr double MIN_BRIDGE_SUPPORT = 0.24;
static constexpr double MIN_CENTER_GAP_SCORE = 0.14;
static constexpr double MAX_FILL_RATIO = 0.62;
static constexpr double MIN_LEG_ROW_CONTINUITY = 0.18;
static constexpr double MIN_BRIDGE_ROW_CONTINUITY = 0.10;

// Numerically equivalent to PureH_CoreLock_05/pure_h_corelock_policy.hpp.
inline IntegrityResult evaluateIntegrity(const IntegrityMetrics& m) {
    IntegrityResult out;
    if (!m.resolved) {
        out.pass = true;
        out.hardChecked = false;
        out.score = 1.0;
        return out;
    }

    out.hardChecked = true;
    const bool disconnectedBlob =
        !m.bridgeConnectivity &&
        (m.spanningComponentFraction < 0.68 || m.bridgeSupport < 0.35);
    const bool filledBlob =
        m.fillRatio > MAX_FILL_RATIO && m.centerGapScore < 0.28;
    const bool missingHStructure =
        (m.bothSidesSupport < 0.18 && m.bridgeSupport < 0.20) ||
        (m.threeOfFourStemSupport < 0.12 && m.bridgeSupport < 0.24);
    const bool missingThreeStrokePattern =
        m.legRowContinuity < MIN_LEG_ROW_CONTINUITY ||
        m.bridgeRowContinuity < MIN_BRIDGE_ROW_CONTINUITY;

    const bool structurePass =
        !disconnectedBlob && !filledBlob &&
        !missingHStructure && !missingThreeStrokePattern;

    const double spanningScore = clamp01(
        (m.spanningComponentFraction - 0.35) / 0.45);
    const double stemScore = clamp01(
        (m.threeOfFourStemSupport - 0.16) / 0.60);
    const double sideScore = clamp01(
        (m.bothSidesSupport - 0.18) / 0.58);
    const double bridgeScore = clamp01(
        (m.bridgeSupport - 0.14) / 0.56);
    const double gapScore = clamp01(
        (m.centerGapScore - 0.06) / 0.62);
    const double fillScore = clamp01(
        (MAX_FILL_RATIO - m.fillRatio) / 0.42);
    const double connectivityScore = m.bridgeConnectivity ? 1.0 : 0.0;
    const double legRunScore = clamp01(m.legRowContinuity / 0.60);
    const double bridgeRunScore = clamp01(m.bridgeRowContinuity / 0.35);

    out.score = clamp01(
        0.18 * connectivityScore +
        0.14 * spanningScore +
        0.12 * stemScore +
        0.12 * sideScore +
        0.12 * bridgeScore +
        0.07 * gapScore +
        0.05 * fillScore +
        0.12 * legRunScore +
        0.08 * bridgeRunScore);
    out.pass = structurePass;
    return out;
}

inline std::array<double, 3> nestedProbeScales() {
    return {{0.46, 0.36, 0.58}};
}

} // namespace helipad_corelock05
