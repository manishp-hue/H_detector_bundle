#pragma once

#include <algorithm>

namespace pure_h_corelock {

struct IntegrityMetrics
{
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

struct IntegrityResult
{
    bool pass = true;
    bool hardChecked = false;
    double score = 1.0;
};

// Deliberately tolerant. This is not a replacement H detector; it only stops a
// resolved non-H blob from gaining fresh OBSERVED authority after the existing
// Rev11 H verifier has already produced a plausible score.
static constexpr double MIN_SPANNING_COMPONENT = 0.50;
static constexpr double MIN_THREE_OF_FOUR_STEMS = 0.28;
static constexpr double MIN_BOTH_SIDES = 0.30;
static constexpr double MIN_BRIDGE_SUPPORT = 0.24;
static constexpr double MIN_CENTER_GAP_SCORE = 0.14;
static constexpr double MAX_FILL_RATIO = 0.62;
static constexpr double MIN_LEG_ROW_CONTINUITY = 0.18;
static constexpr double MIN_BRIDGE_ROW_CONTINUITY = 0.10;

inline double clamp01(double v)
{
    return std::max(0.0, std::min(1.0, v));
}

inline IntegrityResult evaluateIntegrity(const IntegrityMetrics& m)
{
    IntegrityResult out;

    if (!m.resolved) {
        // Preserve the trusted Rev11 FAR/temporal semantics. Tiny H often lacks
        // enough pixels to prove hard connectivity in one frame.
        out.pass = true;
        out.hardChecked = false;
        out.score = 1.0;
        return out;
    }

    out.hardChecked = true;

    // Negative guard: reject clear non-H evidence, but do not require every
    // projection metric to be perfect. The existing Rev11 H verifier remains
    // the positive authority. This preserves perspective/blur tolerance.
    const bool disconnectedBlob =
        !m.bridgeConnectivity &&
        (m.spanningComponentFraction < 0.68 || m.bridgeSupport < 0.35);
    const bool filledBlob =
        m.fillRatio > MAX_FILL_RATIO &&
        m.centerGapScore < 0.28;
    const bool missingHStructure =
        (m.bothSidesSupport < 0.18 && m.bridgeSupport < 0.20) ||
        (m.threeOfFourStemSupport < 0.12 && m.bridgeSupport < 0.24);
    const bool missingThreeStrokePattern =
        m.legRowContinuity < MIN_LEG_ROW_CONTINUITY ||
        m.bridgeRowContinuity < MIN_BRIDGE_ROW_CONTINUITY;

    const bool structurePass =
        !disconnectedBlob &&
        !filledBlob &&
        !missingHStructure &&
        !missingThreeStrokePattern;

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
    const double legRunScore = clamp01(
        m.legRowContinuity / 0.60);
    const double bridgeRunScore = clamp01(
        m.bridgeRowContinuity / 0.35);

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

} // namespace pure_h_corelock
