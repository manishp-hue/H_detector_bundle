#pragma once

#include <algorithm>

namespace pure_h_clearance {

// Pure policy layer for the Rev11 H-only clearance gate.  Image processing
// supplies normalized residual ratios; this file decides whether those
// residuals represent meaningful clutter.  FAR H is intentionally not hard
// rejected here because Rev11's temporal/far evidence remains authoritative.
struct Metrics
{
    bool resolved = false;
    double internalClutterScore = 1.0;
    double bodyResidualRatio = 0.0;
    double bodyLargestComponentRatio = 0.0;
    double haloResidualRatio = 0.0;
    double haloLargestComponentRatio = 0.0;
};

struct Result
{
    bool pass = true;
    bool hardChecked = false;
    bool bodyClean = true;
    bool haloClean = true;
    double score = 1.0;
};

static constexpr double BODY_RESIDUAL_MAX = 0.12;
static constexpr double BODY_LARGEST_COMPONENT_MAX = 0.065;
static constexpr double HALO_RESIDUAL_MAX = 0.08;
static constexpr double HALO_LARGEST_COMPONENT_MAX = 0.055;
static constexpr double STRICT_INTERNAL_CLUTTER_MIN = 0.50;

inline double clamp01(double v)
{
    return std::max(0.0, std::min(1.0, v));
}

inline Result evaluate(const Metrics& m)
{
    Result out;

    if (!m.resolved) {
        // Tiny/far H may have only fractional-pixel clearance.  Preserve the
        // trusted Rev11 FAR/temporal decision and use this only as a soft score.
        out.pass = true;
        out.hardChecked = false;
        out.bodyClean = true;
        out.haloClean = true;
        out.score = clamp01(0.65 + 0.35 * m.internalClutterScore);
        return out;
    }

    out.hardChecked = true;
    out.bodyClean =
        m.bodyResidualRatio <= BODY_RESIDUAL_MAX &&
        m.bodyLargestComponentRatio <= BODY_LARGEST_COMPONENT_MAX;

    out.haloClean =
        m.haloResidualRatio <= HALO_RESIDUAL_MAX &&
        m.haloLargestComponentRatio <= HALO_LARGEST_COMPONENT_MAX;

    const double bodyResidualScore = clamp01(
        1.0 - m.bodyResidualRatio / BODY_RESIDUAL_MAX);
    const double bodyComponentScore = clamp01(
        1.0 - m.bodyLargestComponentRatio / BODY_LARGEST_COMPONENT_MAX);
    const double haloResidualScore = clamp01(
        1.0 - m.haloResidualRatio / HALO_RESIDUAL_MAX);
    const double haloComponentScore = clamp01(
        1.0 - m.haloLargestComponentRatio / HALO_LARGEST_COMPONENT_MAX);
    out.score = clamp01(
        0.32 * bodyResidualScore +
        0.23 * bodyComponentScore +
        0.25 * haloResidualScore +
        0.20 * haloComponentScore);

    out.pass = out.bodyClean && out.haloClean;
    return out;
}

} // namespace pure_h_clearance
