#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace corelock05_policy {

enum class FarSignatureKind
{
    Strong = 0,
    Ambiguous = 1,
    Reject = 2
};

struct FarTopologyMetrics
{
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

struct FarTopologyDecision
{
    FarSignatureKind kind = FarSignatureKind::Reject;
    double score = 0.0;
};

inline double clamp01(double v)
{
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

inline double min4(double a, double b, double c, double d)
{
    return std::min(std::min(a, b), std::min(c, d));
}

inline FarTopologyDecision classifyFarTopology(const FarTopologyMetrics& m)
{
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

inline double orientationDistance180(double a, double b)
{
    double d = std::fmod(std::abs(a - b), 180.0);
    if (d > 90.0) d = 180.0 - d;
    return d;
}

inline bool farSignaturesCompatible(
    const FarTopologyMetrics& a,
    bool aDarkPolarity,
    double aOrientationDeg,
    const FarTopologyMetrics& b,
    bool bDarkPolarity,
    double bOrientationDeg)
{
    if (aDarkPolarity != bDarkPolarity) return false;
    if (std::abs(a.bridgeY - b.bridgeY) > 0.16) return false;
    if (orientationDistance180(aOrientationDeg, bOrientationDeg) > 28.0) return false;

    const double aTop = 0.5 * (a.leftTop + a.rightTop);
    const double aBottom = 0.5 * (a.leftBottom + a.rightBottom);
    const double bTop = 0.5 * (b.leftTop + b.rightTop);
    const double bBottom = 0.5 * (b.leftBottom + b.rightBottom);
    if (std::abs(aTop - bTop) > 0.42) return false;
    if (std::abs(aBottom - bBottom) > 0.42) return false;
    return true;
}

inline bool farSignatureConsensusReady(
    FarSignatureKind kind,
    int compatibleCount)
{
    if (kind == FarSignatureKind::Reject) return false;
    if (kind == FarSignatureKind::Strong) return true;
    return compatibleCount >= 2;
}

inline bool shouldStartProvisional(bool farMode, FarSignatureKind kind)
{
    return farMode && kind == FarSignatureKind::Ambiguous;
}

inline bool provisionalPromoted(int freshCompatible, int required = 2)
{
    return freshCompatible >= required;
}

inline bool provisionalPredictionAllowed(int priorConsecutiveMisses)
{
    return priorConsecutiveMisses <= 0;
}

inline bool shouldAttemptNestedProbe(
    bool resolvedProposal,
    bool normalShapeReady,
    bool componentRefineSucceeded)
{
    return resolvedProposal && !normalShapeReady && !componentRefineSucceeded;
}

inline std::array<double, 3> nestedProbeScales()
{
    return {{0.46, 0.36, 0.58}};
}

} // namespace corelock05_policy
