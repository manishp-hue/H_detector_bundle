#pragma once

namespace corelock04_policy {

struct FarMicroMetrics
{
    double sideSupport = 0.0;
    double bridgeSupport = 0.0;
    double gapEvidence = 0.0;
    double blobCompactness = 1.0;
    double largestComponentFraction = 0.0;
    double twoSideRowContinuity = 0.0;
};

struct FarMicroDecision
{
    bool pass = false;
    bool ambiguous = true;
    double score = 0.0;
};

inline double clamp01(double v)
{
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

inline FarMicroDecision evaluateFarMicro(const FarMicroMetrics& m)
{
    FarMicroDecision out;

    const double antiBlob = clamp01((0.78 - m.blobCompactness) / 0.42);
    out.score = clamp01(
        0.25 * m.sideSupport +
        0.24 * m.bridgeSupport +
        0.16 * m.gapEvidence +
        0.13 * antiBlob +
        0.12 * m.largestComponentFraction +
        0.10 * m.twoSideRowContinuity);

    const bool obviousBlob =
        m.blobCompactness >= 0.76 &&
        (m.gapEvidence < 0.12 || m.bridgeSupport < 0.16);

    const bool missingHMicroStructure =
        m.sideSupport < 0.16 ||
        m.bridgeSupport < 0.12 ||
        m.gapEvidence < 0.05;
    const bool structurallyCoherent =
        m.largestComponentFraction >= 0.55 ||
        m.twoSideRowContinuity >= 0.45;

    out.pass = !obviousBlob && !missingHMicroStructure &&
        structurallyCoherent && out.score >= 0.25;

    // Strong tiny H evidence stays on the existing Rev11 confirmation path.
    // Only marginal-but-plausible candidates require extra compatible evidence.
    out.ambiguous =
        !out.pass ||
        out.score < 0.50 ||
        m.blobCompactness > 0.58 ||
        m.gapEvidence < 0.20 ||
        m.largestComponentFraction < 0.55 ||
        m.twoSideRowContinuity < 0.45;

    return out;
}

inline bool requiresFarMicroConsensus(const FarMicroDecision& d)
{
    return d.pass && d.ambiguous;
}

inline bool farMicroConsensusReady(bool ambiguous, int compatibleCount)
{
    return !ambiguous || compatibleCount >= 2;
}

inline bool shouldCanonicalizeIdentity(
    bool resolved,
    bool supportHeavy,
    bool scaleMismatch)
{
    return resolved && (supportHeavy || scaleMismatch);
}

inline bool escapeArmed(int directMissStreak, int minDirectMisses = 2)
{
    return directMissStreak >= minDirectMisses;
}

inline bool escapeReady(
    int directMissStreak,
    bool independentlyVerified,
    bool resolvedIntegrityPass,
    int persistentChecks,
    int minDirectMisses = 2,
    int requiredPersistentChecks = 2)
{
    return escapeArmed(directMissStreak, minDirectMisses) &&
           independentlyVerified &&
           resolvedIntegrityPass &&
           persistentChecks >= requiredPersistentChecks;
}

} // namespace corelock04_policy
