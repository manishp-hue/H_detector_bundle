#pragma once

namespace corelock03_policy {

// Preserve Rev11's aligned FAR-temporal authority path. Every ordinary
// current-frame recovery candidate must be re-verified locally before it can
// refresh the active target as OBSERVED.
inline bool requiresRecoveryReverify(
    bool farTrack,
    bool temporalSupported)
{
    return !(farTrack && temporalSupported);
}

// The normal direct H verification remains the cheap real-field fast path.
// Core refinement is allowed only for a resolved target after a genuine direct
// miss / integrity failure or when the measured foreground is much smaller than
// the predicted outer target scale.
inline bool shouldAttemptDirectCoreRefine(
    bool resolved,
    bool shapeReady,
    bool scaleMismatch)
{
    return resolved && (!shapeReady || scaleMismatch);
}

} // namespace corelock03_policy
